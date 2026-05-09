"""Measure the eval-CE floor when the activation injection is silenced.

Loads a trained AV, then:
  1. Eval CE WITH activation: normal forward.
  2. Eval CE WITH activation set to 0: zero out h_l before injection.

If (1) ≈ (2), the activation contributes ~0 to the loss reduction — the model
just learned P(summary | prompt prior). If (1) < (2) by a meaningful margin,
the activation IS contributing but plateaus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from nla.qwen import load_qwen, best_amp_dtype
from nla.av import AV
from nla.data_phase3 import AVDataset, av_collate, load_phase3_inputs
from nla.splits import make_or_load_split


@torch.no_grad()
def ce_eval(av, loader, device, *, zero_h: bool = False, max_batches: int = 50):
    av.eval()
    total, n = 0.0, 0
    dtype = best_amp_dtype()
    for batch_i, batch in enumerate(loader):
        if batch_i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        h = batch["h"].to(device)
        if zero_h:
            h = torch.zeros_like(h)
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            out = av(input_ids=ids, attention_mask=attn, h_l=h, labels=labels)
        total += float(out.loss); n += 1
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--raw-h", action="store_true")
    p.add_argument("--layer", default="h3")
    p.add_argument("--max-batches", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda")
    base, tok, act_id = load_qwen(use_4bit=True, device_map="auto")
    av = AV(base, act_id).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("trainable_state", {})
    av.load_state_dict(state, strict=False)
    print(f"loaded {len(state)} keys (step={ckpt.get('step')})")

    layer_key = args.layer if args.raw_h else args.layer + "_l2"
    corpus, summaries, h = load_phase3_inputs(layer_key=layer_key)
    train_idx, eval_idx = make_or_load_split(len(corpus))

    ds = AVDataset(eval_idx, summaries, h, tok)
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        collate_fn=lambda b: av_collate(b, tok.pad_token_id))

    ce_with = ce_eval(av, loader, device, zero_h=False, max_batches=args.max_batches)
    ce_zero = ce_eval(av, loader, device, zero_h=True, max_batches=args.max_batches)
    delta = ce_zero - ce_with

    print(f"\n=== eval CE on {args.max_batches} batches × 8 = up to {args.max_batches * 8} rows ===")
    print(f"  CE WITH    activation : {ce_with:.4f}")
    print(f"  CE WITHOUT activation : {ce_zero:.4f}  (h zeroed)")
    print(f"  delta                 : {delta:+.4f}")
    if delta < 0.01:
        print("  → activation contributes ~0; model is at prompt-prior floor")
    elif delta < 0.10:
        print("  → activation contributes some but is heavily under-utilized")
    else:
        print("  → activation contributes meaningfully")


if __name__ == "__main__":
    main()
