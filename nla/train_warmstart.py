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


def _trainable_keys(state_dict: dict) -> dict:
    """Subset of state_dict that we actually train (LoRA adapters + proj/head)."""
    return {
        k: v.detach().cpu()
        for k, v in state_dict.items()
        if any(t in k for t in ("lora_", ".proj.", ".head."))
    }


def _save_checkpoint(module: nn.Module, save_dir: Path, step: int, tag: str = "ar_or_av"):
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "trainable_state": _trainable_keys(module.state_dict())},
               save_dir / f"step_{step}.pt")


def save_final(module: nn.Module, save_dir: Path, step: int, *, history: dict | None = None,
                stop_reason: str | None = None) -> Path:
    """Persist a 'final.pt' that resume-on-second-run looks for."""
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "final.pt"
    torch.save({
        "step": step,
        "stop_reason": stop_reason,
        "history": history,
        "trainable_state": _trainable_keys(module.state_dict()),
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
) -> dict:
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    av.train()

    opt = torch.optim.AdamW(_split_param_groups(av, lr_lora, lr_proj, weight_decay))
    stopper = EarlyStopper(mode="ce", patience=patience, min_delta=min_delta, min_steps=min_steps)
    history = []
    t0 = time.time()
    step = 0
    accum = 0
    train_iter = iter(train_loader)
    progress = tqdm(total=max_steps, desc="AV")

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

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            out = av(input_ids=input_ids, attention_mask=attn, h_l=h, labels=labels)
            loss = out.loss / grad_accum
        loss.backward()
        accum += 1
        if accum >= grad_accum:
            torch.nn.utils.clip_grad_norm_([p for p in av.parameters() if p.requires_grad], grad_clip)
            _set_lrs(opt, max_steps=max_steps, step=step, lr_lora=lr_lora, lr_proj=lr_proj,
                     warmup_steps=warmup_steps)
            opt.step(); opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            progress.update(1)

            if step % log_every == 0:
                history.append({"step": step, "train_loss": float(loss.item() * grad_accum),
                                "elapsed_sec": time.time() - t0})
            if step % eval_every == 0:
                ce = _ce_eval_av(av, eval_loader, device)
                stopper.update(ce, step)
                history.append({"step": step, "eval_ce": ce, "elapsed_sec": time.time() - t0})
                progress.set_postfix(eval_ce=f"{ce:.4f}")
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
    step = 0
    accum = 0
    train_iter = iter(train_loader)
    progress = tqdm(total=max_steps, desc="AR")

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

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            h_hat = ar(input_ids=input_ids, attention_mask=attn)
            loss = ((h - h_hat) ** 2).sum(dim=-1).mean() / grad_accum
        loss.backward()
        accum += 1
        if accum >= grad_accum:
            torch.nn.utils.clip_grad_norm_([p for p in ar.parameters() if p.requires_grad], grad_clip)
            _set_lrs(opt, max_steps=max_steps, step=step, lr_lora=lr_lora, lr_proj=lr_proj,
                     warmup_steps=warmup_steps)
            opt.step(); opt.zero_grad(set_to_none=True)
            accum = 0
            step += 1
            progress.update(1)

            if step % log_every == 0:
                history.append({"step": step, "train_mse": float(loss.item() * grad_accum),
                                "elapsed_sec": time.time() - t0})
            if step % eval_every == 0:
                mse = _mse_eval_ar(ar, eval_loader, device)
                ev: dict = {"step": step, "eval_mse": mse, "elapsed_sec": time.time() - t0}
                if use_fve:
                    fve = fve_eval_fn(step)
                    ev["fve"] = fve
                    stopper.update(fve, step)
                    progress.set_postfix(mse=f"{mse:.4f}", fve=f"{fve:.3f}")
                else:
                    stopper.update(mse, step)
                    progress.set_postfix(mse=f"{mse:.4f}")
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
