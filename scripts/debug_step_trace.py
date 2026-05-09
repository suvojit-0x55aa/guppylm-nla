"""Train AR for 20 steps and report loss + grad norms + param-norm at each step.
First step where loss → NaN tells us exactly which optimizer step triggered the blowup.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from nla.qwen import load_qwen, best_amp_dtype
from nla.av import AV
from nla.ar import AR
from nla.data_phase3 import (
    AVDataset, ARDataset, av_collate, ar_collate, load_phase3_inputs,
)
from nla.train_warmstart import _split_param_groups, _set_lrs


def fnorm(t):
    if not torch.is_tensor(t):
        return float("nan")
    return float(t.detach().float().norm())


def has_nan(t):
    if t is None or not torch.is_tensor(t):
        return False
    if not t.dtype.is_floating_point:
        return False
    return bool(torch.isnan(t).any().item()) or bool(torch.isinf(t).any().item())


def main():
    device = torch.device("cuda")
    dtype = best_amp_dtype()
    print(f"device={device} amp_dtype={dtype}")

    base, tok, act_id = load_qwen(use_4bit=True, device_map="auto")
    av = AV(base, act_id).to(device)
    ar = AR(av.base, d_substrate=384).to(device)

    corpus, summaries, h = load_phase3_inputs()

    # AR train, batch=4
    ar_ds = ARDataset(list(range(64)), summaries, h, tok)
    loader = DataLoader(ar_ds, batch_size=4, shuffle=False,
                        collate_fn=lambda b: ar_collate(b, tok.pad_token_id))

    opt = torch.optim.AdamW(_split_param_groups(ar, 2e-4, 1e-3, 0.01))
    grad_clip = 1.0

    print(f"\n== ar.head.weight init ==")
    print(f"  norm={fnorm(ar.head.weight):.4g} dtype={ar.head.weight.dtype}")
    # Activate AR adapter so its LoRA params have requires_grad=True.
    ar.base.set_adapter("ar")
    # List all trainable param names.
    trainable = [(n, p) for n, p in ar.named_parameters() if p.requires_grad]
    print(f"  trainable count={len(trainable)}")
    print(f"  first 5 trainable: {[n for n, _ in trainable[:5]]}")
    print(f"  any 'ar.' in names: {[n for n, _ in trainable if '.ar.' in n][:3]}")
    print(f"  any 'av.' in names: {[n for n, _ in trainable if '.av.' in n][:3]}")
    lora_param = None
    lora_name = None
    for n, p in trainable:
        if "lora_" in n:
            lora_param = p
            lora_name = n
            break
    print(f"  lora monitor: {lora_name}  norm={fnorm(lora_param) if lora_param is not None else 'NONE'}")

    print(f"\n step | loss        | head_grad   | head_w     | lora_grad   | lora_w     | clipped")
    print(f"------|-------------|-------------|------------|-------------|------------|--------")
    ar.train()
    step = 0
    train_iter = iter(loader)
    while step < 20:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        h_b = batch["h"].to(device)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            h_hat = ar(input_ids=input_ids, attention_mask=attn)
            loss = ((h_b - h_hat) ** 2).sum(dim=-1).mean()

        loss.backward()
        head_g = fnorm(ar.head.weight.grad)
        lora_g = (
            fnorm(lora_param.grad) if (lora_param is not None and lora_param.grad is not None)
            else float("nan")
        )
        # Clip grads
        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in ar.parameters() if p.requires_grad], grad_clip
        )
        _set_lrs(opt, max_steps=10000, step=step, lr_lora=2e-4, lr_proj=1e-3, warmup_steps=200)
        opt.step()
        step += 1

        loss_v = float(loss.item())
        flag = " ⚠" if (np.isnan(loss_v) or np.isinf(loss_v)) else ""
        head_nan = has_nan(ar.head.weight)
        lora_nan = has_nan(lora_param)
        print(f"  {step:3d} | {loss_v:+11.4g}{flag} | {head_g:11.4g} | {fnorm(ar.head.weight):10.4g} | "
              f"{lora_g:11.4g} | {fnorm(lora_param):10.4g} | total={float(total_norm):.3g}"
              + (" head=NaN" if head_nan else "")
              + (" lora=NaN" if lora_nan else ""))

        if np.isnan(loss_v) or np.isinf(loss_v) or head_nan or lora_nan:
            print(f"\n  STOPPED: NaN/Inf detected at step {step}")
            print(f"  ar.head.weight contains NaN: {has_nan(ar.head.weight)}")
            print(f"  ar.head.weight stats: norm={fnorm(ar.head.weight):.4g}")
            tf = ar.head.weight.detach().float()
            n_nan = int(torch.isnan(tf).sum().item())
            n_inf = int(torch.isinf(tf).sum().item())
            print(f"    n_nan={n_nan} / {tf.numel()} ; n_inf={n_inf}")
            break


if __name__ == "__main__":
    main()
