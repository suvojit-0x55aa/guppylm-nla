"""Phase 3 warm-start training loops for AV and AR.

Both loops share:
  - AdamW with two param groups (LoRA at lr_lora, projection/head at lr_proj).
  - 200-step warmup → cosine decay to 0 over max_steps.
  - fp16 autocast for compute, fp32 for losses.
  - Eval every `eval_every` steps; checkpoint every `ckpt_every`.
  - Convergence-based stopping (`EarlyStopper`) + wall-clock budget cap.

Stop reasons returned: "converged" | "max_steps" | "time_budget".
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .qwen import best_amp_dtype


# ── Early stopping ────────────────────────────────────────────────────────────


@dataclass
class EarlyStopper:
    mode: str                 # "ce" (lower better, relative) or "fve" (higher better, absolute)
    patience: int = 3
    min_delta: float = 0.005
    min_steps: int = 1000
    history: list[float] = None
    last_step_recorded: int = 0

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def update(self, value: float, step: int) -> None:
        self.history.append(value)
        self.last_step_recorded = step

    def should_stop(self, step: int) -> bool:
        if step < self.min_steps:
            return False
        if len(self.history) < self.patience + 1:
            return False
        # Compare last `patience` values to the value `patience` evals ago.
        baseline = self.history[-self.patience - 1]
        recent = self.history[-self.patience:]
        if self.mode == "ce":
            # Loss: stop if relative drop < min_delta for all recent evals.
            if baseline <= 0:
                return False
            improvements = [(baseline - v) / baseline for v in recent]
            return all(d < self.min_delta for d in improvements)
        elif self.mode == "fve":
            # FVE: bounded; stop if no improvement of >= min_delta for any recent eval.
            return all((v - baseline) < self.min_delta for v in recent)
        else:
            raise ValueError(f"unknown mode {self.mode}")


# ── Schedulers / utilities ────────────────────────────────────────────────────


def cosine_warmup_lr(step: int, *, max_steps: int, warmup_steps: int = 200,
                     base_lr: float, min_lr: float = 0.0) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _split_param_groups(module: nn.Module, lr_lora: float, lr_proj: float, wd: float):
    """Two groups: LoRA-suffixed (lora_A/lora_B) at lr_lora, the rest at lr_proj.
    Both groups apply weight decay; biases and LayerNorm-style 1D weights are
    excluded from wd."""
    lora_decay, lora_no_decay = [], []
    proj_decay, proj_no_decay = [], []
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        is_lora = "lora_" in name
        is_no_decay = p.ndim < 2 or name.endswith(".bias")
        bucket = (lora_no_decay if is_no_decay else lora_decay) if is_lora \
            else (proj_no_decay if is_no_decay else proj_decay)
        bucket.append(p)
    return [
        {"params": lora_decay,    "lr": lr_lora, "weight_decay": wd},
        {"params": lora_no_decay, "lr": lr_lora, "weight_decay": 0.0},
        {"params": proj_decay,    "lr": lr_proj, "weight_decay": wd},
        {"params": proj_no_decay, "lr": lr_proj, "weight_decay": 0.0},
    ]


def _set_lrs(opt: torch.optim.Optimizer, *, max_steps: int, step: int,
             lr_lora: float, lr_proj: float, warmup_steps: int = 200):
    lora_lr = cosine_warmup_lr(step, max_steps=max_steps, warmup_steps=warmup_steps, base_lr=lr_lora)
    proj_lr = cosine_warmup_lr(step, max_steps=max_steps, warmup_steps=warmup_steps, base_lr=lr_proj)
    for g in opt.param_groups[:2]:
        g["lr"] = lora_lr
    for g in opt.param_groups[2:]:
        g["lr"] = proj_lr
    return lora_lr, proj_lr


def _trainable_state(module: nn.Module) -> dict:
    """Subset of state_dict containing only trainable parameters
    (LoRA adapters + proj/head). Uses requires_grad rather than name-matching
    so we don't silently drop the projection/head when their key paths differ
    from a hardcoded expectation."""
    trainable_names = {n for n, p in module.named_parameters() if p.requires_grad}
    out: dict = {}
    for k, v in module.state_dict().items():
        # state_dict keys can include a "module." prefix or other wrappers; map
        # back by suffix-match against named_parameters() names.
        if k in trainable_names or any(k.endswith(name) for name in trainable_names):
            out[k] = v.detach().cpu()
    return out


def _save_checkpoint(module: nn.Module, save_dir: Path, step: int, tag: str = "ar_or_av",
                     keep_last: int = 3):
    """Save step_<n>.pt; rotate to keep only the last `keep_last` step files.
    final.pt + best.pt are never rotated (different filename pattern).
    With each ckpt at ~1.3 GB on the resized-embedding base, keeping all of a
    6000-step run would need ~33 GB; we cap at ~4 GB."""
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "trainable_state": _trainable_state(module)},
               save_dir / f"step_{step}.pt")
    step_files = sorted(save_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    for old in step_files[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass


def _save_best(module: nn.Module, save_dir: Path, step: int, metric_name: str,
               metric_value: float):
    """Persist the best-so-far snapshot to best.pt. Filename does not match
    step_*.pt so the rotation in _save_checkpoint never touches it."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "trainable_state": _trainable_state(module),
    }, save_dir / "best.pt")


def save_final(module: nn.Module, save_dir: Path, step: int, *, history: dict | None = None,
                stop_reason: str | None = None) -> Path:
    """Persist a 'final.pt' that resume-on-second-run looks for."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "final.pt"
    state = _trainable_state(module)
    torch.save({
        "step": step,
        "stop_reason": stop_reason,
        "history": history,
        "trainable_state": state,
        "n_keys": len(state),
    }, final_path)
    return final_path


def load_final(module: nn.Module, save_dir: Path, device: torch.device | str = "cuda") -> dict | None:
    """If save_dir/final.pt exists, load its trainable_state into module. Returns the
    loaded checkpoint metadata (without trainable_state) or None."""
    final_path = Path(save_dir) / "final.pt"
    if not final_path.exists():
        return None
    ckpt = torch.load(final_path, map_location=device, weights_only=False)
    state = ckpt.get("trainable_state", {})
    missing, unexpected = module.load_state_dict(state, strict=False)
    return {"step": ckpt.get("step"), "stop_reason": ckpt.get("stop_reason"),
            "history": ckpt.get("history", []), "n_loaded_keys": len(state),
            "n_unexpected": len(unexpected)}


def load_latest_checkpoint(
    module: nn.Module, save_dir: Path, device: torch.device | str = "cuda",
) -> dict | None:
    """Load the highest-step checkpoint in save_dir for partial-progress resume.
    Prefers final.pt; otherwise the highest step_<n>.pt. Returns metadata + step.
    Optimizer state is NOT preserved (causes a ~50-step loss bump until AdamW
    re-warms — cheaper than re-running 500 steps from scratch)."""
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    final_path = save_dir / "final.pt"
    if final_path.exists():
        return load_final(module, save_dir, device=device)
    step_files = sorted(save_dir.glob("step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not step_files:
        return None
    latest = step_files[-1]
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    state = ckpt.get("trainable_state", {})
    missing, unexpected = module.load_state_dict(state, strict=False)
    return {"step": ckpt.get("step", int(latest.stem.split("_")[1])),
            "stop_reason": "resumed",
            "history": [], "n_loaded_keys": len(state),
            "n_unexpected": len(unexpected),
            "from_path": str(latest)}


# ── AV training ───────────────────────────────────────────────────────────────


def _ce_eval_av(av: nn.Module, eval_loader: DataLoader, device, max_batches: int = 50) -> float:
    av.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch_i, batch in enumerate(eval_loader):
            if batch_i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            h = batch["h"].to(device)
            out = av(input_ids=input_ids, attention_mask=attn, h_l=h, labels=labels)
            total_loss += float(out.loss); n += 1
    av.train()
    return total_loss / max(n, 1)


def train_av(
    av: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    max_steps: int = 10_000,
    min_steps: int = 1000,
    eval_every: int = 200,
    ckpt_every: int = 500,
    grad_accum: int = 4,
    grad_clip: float = 1.0,
    lr_lora: float = 2e-4,
    lr_proj: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    patience: int = 3,
    min_delta: float = 0.005,
    time_budget_sec: float = 5400.0,                    # 1.5 hr default
    save_dir: str | Path = "checkpoints/phase3/av",
    device: torch.device | str = "cuda",
    log_every: int = 50,
    start_step: int = 0,                                # resume cursor (weights restored externally)
) -> dict:
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    av.train()

    opt = torch.optim.AdamW(_split_param_groups(av, lr_lora, lr_proj, weight_decay))
    stopper = EarlyStopper(mode="ce", patience=patience, min_delta=min_delta, min_steps=min_steps)
    history = []
    t0 = time.time()
    step = start_step
    accum = 0
    train_iter = iter(train_loader)
    progress = tqdm(total=max_steps, initial=step, desc="AV")
    # Live stats — exponential-moving average of loss, last grad-norm/LR/eval, GPU mem.
    loss_ema = None
    last_gn = float("nan")
    last_lr = float("nan")
    last_eval_ce = float("nan")
    best_eval_ce = float("inf")

    stop_reason = "max_steps"
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        h = batch["h"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=best_amp_dtype()):
            out = av(input_ids=input_ids, attention_mask=attn, h_l=h, labels=labels)
            loss = out.loss / grad_accum
        loss.backward()
        accum += 1
        if accum >= grad_accum:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in av.parameters() if p.requires_grad], grad_clip)
            last_gn = float(gn) if torch.is_tensor(gn) else float(gn)
            lora_lr, proj_lr = _set_lrs(opt, max_steps=max_steps, step=step,
                                        lr_lora=lr_lora, lr_proj=lr_proj,
                                        warmup_steps=warmup_steps)
            last_lr = lora_lr
            opt.step(); opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            progress.update(1)
            train_loss_full = float(loss.item()) * grad_accum
            loss_ema = train_loss_full if loss_ema is None else 0.98 * loss_ema + 0.02 * train_loss_full
            gpu_gb = (torch.cuda.memory_reserved() / 1024 ** 3) if device.type == "cuda" else 0.0
            progress.set_postfix(
                loss=f"{loss_ema:.3f}", gn=f"{last_gn:.2f}", lr=f"{last_lr:.1e}",
                eval=f"{last_eval_ce:.3f}", gpu=f"{gpu_gb:.1f}G",
            )

            if step % log_every == 0:
                history.append({"step": step, "train_loss": float(loss.item() * grad_accum),
                                "elapsed_sec": time.time() - t0})
            if step % eval_every == 0:
                ce = _ce_eval_av(av, eval_loader, device)
                stopper.update(ce, step)
                history.append({"step": step, "eval_ce": ce, "elapsed_sec": time.time() - t0})
                last_eval_ce = ce
                if ce < best_eval_ce:
                    best_eval_ce = ce
                    _save_best(av, save_dir, step, "eval_ce", ce)
                if stopper.should_stop(step):
                    stop_reason = "converged"
                    break
            if step % ckpt_every == 0:
                _save_checkpoint(av, save_dir, step, tag="av")
            if time.time() - t0 > time_budget_sec:
                stop_reason = "time_budget"
                break

    progress.close()
    _save_checkpoint(av, save_dir, step, tag="av")
    elapsed = time.time() - t0
    result = {"stop_reason": stop_reason, "final_step": step, "elapsed_sec": elapsed, "history": history}
    save_final(av, save_dir, step, history=history, stop_reason=stop_reason)
    return result


# ── AR training ───────────────────────────────────────────────────────────────


def _mse_eval_ar(ar: nn.Module, eval_loader: DataLoader, device, max_batches: int = 50) -> float:
    ar.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch_i, batch in enumerate(eval_loader):
            if batch_i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            h = batch["h"].to(device)
            h_hat = ar(input_ids=input_ids, attention_mask=attn)
            total += float(((h - h_hat) ** 2).sum(dim=-1).mean().item()); n += 1
    ar.train()
    return total / max(n, 1)


def train_ar(
    ar: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    *,
    fve_eval_fn: Optional[Callable[[int], float]] = None,
    max_steps: int = 10_000,
    min_steps: int = 1000,
    eval_every: int = 200,
    ckpt_every: int = 500,
    grad_accum: int = 4,
    grad_clip: float = 1.0,
    lr_lora: float = 2e-4,
    lr_proj: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    patience: int = 3,
    min_delta: float = 0.005,
    time_budget_sec: float = 5400.0,
    save_dir: str | Path = "checkpoints/phase3/ar",
    device: torch.device | str = "cuda",
    log_every: int = 50,
    start_step: int = 0,                                # resume cursor (weights restored externally)
) -> dict:
    """Trains AR. If `fve_eval_fn(step)` is provided, FVE is computed at every
    eval and used as the early-stop signal; otherwise raw MSE is used."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    ar.train()

    opt = torch.optim.AdamW(_split_param_groups(ar, lr_lora, lr_proj, weight_decay))
    use_fve = fve_eval_fn is not None
    stopper = EarlyStopper(
        mode="fve" if use_fve else "ce",
        patience=patience, min_delta=min_delta, min_steps=min_steps,
    )
    history = []
    t0 = time.time()
    step = start_step
    accum = 0
    train_iter = iter(train_loader)
    progress = tqdm(total=max_steps, initial=step, desc="AR")
    # Live stats — EMA loss, last grad-norm/LR/eval-mse/fve, GPU mem.
    loss_ema = None
    last_gn = float("nan")
    last_lr = float("nan")
    last_eval_mse = float("nan")
    last_fve = float("nan")
    best_fve = float("-inf")
    best_mse = float("inf")

    stop_reason = "max_steps"
    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        h = batch["h"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=best_amp_dtype()):
            h_hat = ar(input_ids=input_ids, attention_mask=attn)
            loss = ((h - h_hat) ** 2).sum(dim=-1).mean() / grad_accum
        loss.backward()
        accum += 1
        if accum >= grad_accum:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in ar.parameters() if p.requires_grad], grad_clip)
            last_gn = float(gn) if torch.is_tensor(gn) else float(gn)
            lora_lr, proj_lr = _set_lrs(opt, max_steps=max_steps, step=step,
                                        lr_lora=lr_lora, lr_proj=lr_proj,
                                        warmup_steps=warmup_steps)
            last_lr = lora_lr
            opt.step(); opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            progress.update(1)
            train_mse_full = float(loss.item()) * grad_accum
            loss_ema = train_mse_full if loss_ema is None else 0.98 * loss_ema + 0.02 * train_mse_full
            gpu_gb = (torch.cuda.memory_reserved() / 1024 ** 3) if device.type == "cuda" else 0.0
            progress.set_postfix(
                mse=f"{loss_ema:.3f}", gn=f"{last_gn:.2f}", lr=f"{last_lr:.1e}",
                emse=f"{last_eval_mse:.3f}", fve=f"{last_fve:.3f}", gpu=f"{gpu_gb:.1f}G",
            )

            if step % log_every == 0:
                history.append({"step": step, "train_mse": float(loss.item() * grad_accum),
                                "elapsed_sec": time.time() - t0})
            if step % eval_every == 0:
                mse = _mse_eval_ar(ar, eval_loader, device)
                ev: dict = {"step": step, "eval_mse": mse, "elapsed_sec": time.time() - t0}
                last_eval_mse = mse
                if mse < best_mse:
                    best_mse = mse
                    if not use_fve:
                        _save_best(ar, save_dir, step, "eval_mse", mse)
                if use_fve:
                    fve = fve_eval_fn(step)
                    ev["fve"] = fve
                    last_fve = fve
                    stopper.update(fve, step)
                    if fve > best_fve:
                        best_fve = fve
                        _save_best(ar, save_dir, step, "fve", fve)
                else:
                    stopper.update(mse, step)
                history.append(ev)
                if stopper.should_stop(step):
                    stop_reason = "converged"
                    break
            if step % ckpt_every == 0:
                _save_checkpoint(ar, save_dir, step, tag="ar")
            if time.time() - t0 > time_budget_sec:
                stop_reason = "time_budget"
                break

    progress.close()
    _save_checkpoint(ar, save_dir, step, tag="ar")
    elapsed = time.time() - t0
    result = {"stop_reason": stop_reason, "final_step": step, "elapsed_sec": elapsed, "history": history}
    save_final(ar, save_dir, step, history=history, stop_reason=stop_reason)
    return result
