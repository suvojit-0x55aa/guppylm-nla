"""Trace where NaN appears in the AV/AR forward pass.

Loads a tiny batch (size=2), walks AV forward step-by-step, then AR forward
step-by-step, and prints {min, max, has_nan, has_inf, dtype, shape, norm}
at every intermediate tensor. First tensor with NaN/Inf is the culprit.

Run on pod: cd /workspace/guppylm-nla && uv run python scripts/debug_nan_trace.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from nla.qwen import load_qwen, best_amp_dtype
from nla.av import AV
from nla.ar import AR
from nla.data_phase3 import (
    AVDataset, ARDataset, av_collate, ar_collate, load_phase3_inputs,
)


def stamp(name, t):
    if t is None:
        print(f"  {name:<40s}  None")
        return
    if not torch.is_tensor(t):
        print(f"  {name:<40s}  {type(t).__name__}={t}")
        return
    has_nan = bool(torch.isnan(t).any().item()) if t.dtype.is_floating_point else False
    has_inf = bool(torch.isinf(t).any().item()) if t.dtype.is_floating_point else False
    if t.dtype.is_floating_point:
        tf = t.detach().float()
        mn, mx = float(tf.min()), float(tf.max())
        nm = float(tf.norm())
    else:
        mn, mx, nm = int(t.min()), int(t.max()), float("nan")
    flag = " ⚠ NAN" if has_nan else (" ⚠ INF" if has_inf else "")
    print(f"  {name:<40s}  shape={tuple(t.shape)} dtype={str(t.dtype).replace('torch.',''):<10s} "
          f"min={mn:+.4g} max={mx:+.4g} norm={nm:.4g}{flag}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = best_amp_dtype()
    print(f"device={device} amp_dtype={dtype}")

    print("\n== load_qwen ==")
    base, tok, act_id = load_qwen(use_4bit=True, device_map="auto")
    print(f"  act_token_id={act_id}  vocab_size={len(tok)}  hidden={base.config.hidden_size}")

    # Check the new <ACT> embedding row directly.
    emb_layer = base.get_input_embeddings()
    emb_w = emb_layer.weight
    print(f"  emb shape={tuple(emb_w.shape)} dtype={emb_w.dtype}")
    stamp("emb weight (overall)", emb_w)
    stamp(f"emb row [{act_id}] (<ACT>)", emb_w[act_id])
    # Random rows for comparison.
    stamp("emb row [0]", emb_w[0])
    stamp("emb row [100]", emb_w[100])

    print("\n== AV init ==")
    av = AV(base, act_id).to(device)
    print(f"  AV trainable params: {av.trainable_parameter_count():,}")
    stamp("av.proj.weight", av.proj.weight)
    stamp("av.proj.bias", av.proj.bias)

    print("\n== AR init (shares base) ==")
    ar = AR(av.base, d_substrate=384).to(device)  # share base / peft model
    print(f"  AR trainable params: {ar.trainable_parameter_count():,}")
    stamp("ar.head.weight", ar.head.weight)
    stamp("ar.head.bias", ar.head.bias)

    print("\n== load 4 rows ==")
    corpus, summaries, h = load_phase3_inputs()
    print(f"  rows={len(corpus)}  h shape={h.shape}  h dtype={h.dtype}")
    h_first = h[:4]
    print(f"  h[0:4] norms = {np.linalg.norm(h_first, axis=1)}")
    stamp("h[0:4]", torch.from_numpy(h_first))

    av_ds = AVDataset([0, 1], summaries, h, tok)
    av_batch = av_collate([av_ds[0], av_ds[1]], pad_token_id=tok.pad_token_id)
    av_batch = {k: v.to(device) for k, v in av_batch.items()}
    print(f"  AV batch input_ids shape={av_batch['input_ids'].shape}")
    print(f"  AV labels: -100 count={(av_batch['labels'] == -100).sum().item()}, real={(av_batch['labels'] != -100).sum().item()}")
    print(f"  <ACT> positions: {(av_batch['input_ids'] == act_id).nonzero().tolist()}")

    print("\n== AV forward (no autocast) ==")
    av.eval()
    with torch.no_grad():
        emb_ids = av_batch["input_ids"]
        e0 = av.base.get_input_embeddings()(emb_ids)
        stamp("emb(input_ids)", e0)
        h_cast = av_batch["h"].to(av.proj.weight.dtype)
        stamp("h cast to proj dtype", h_cast)
        inj = av.proj(h_cast)
        stamp("proj(h)", inj)
        inj1 = inj.unsqueeze(1).to(e0.dtype)
        mask = (emb_ids == av.act_id).unsqueeze(-1)
        emb_full = torch.where(mask, inj1.expand_as(e0), e0)
        stamp("emb after injection", emb_full)

        # Forward through base directly (no labels)
        out = av.base(inputs_embeds=emb_full, attention_mask=av_batch["attention_mask"], return_dict=True)
        stamp("base logits", out.logits)
        # Now with labels for CE loss
        out2 = av.base(inputs_embeds=emb_full, attention_mask=av_batch["attention_mask"],
                       labels=av_batch["labels"], return_dict=True)
        stamp("base loss (with labels)", out2.loss)
        stamp("base logits (with labels)", out2.logits)

    print("\n== AV forward WITH autocast (matches train) ==")
    av.train()
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        out3 = av(av_batch["input_ids"], av_batch["attention_mask"], av_batch["h"], labels=av_batch["labels"])
    stamp("autocast loss", out3.loss)
    stamp("autocast logits", out3.logits)

    print("\n== AR forward ==")
    ar_ds = ARDataset([0, 1], summaries, h, tok)
    ar_batch = ar_collate([ar_ds[0], ar_ds[1]], pad_token_id=tok.pad_token_id)
    ar_batch = {k: v.to(device) for k, v in ar_batch.items()}
    print(f"  AR batch input_ids shape={ar_batch['input_ids'].shape}")
    ar.eval()
    with torch.no_grad():
        h_hat = ar(ar_batch["input_ids"], ar_batch["attention_mask"])
    stamp("AR h_hat (eval)", h_hat)

    ar.train()
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        h_hat_t = ar(ar_batch["input_ids"], ar_batch["attention_mask"])
    stamp("AR h_hat (autocast)", h_hat_t)

    print("\n== AV: 1 backward step on real data ==")
    av.train()
    av.zero_grad()
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        out4 = av(av_batch["input_ids"], av_batch["attention_mask"], av_batch["h"], labels=av_batch["labels"])
        loss = out4.loss
    stamp("loss (autocast)", loss)
    if torch.isfinite(loss):
        loss.backward()
        # check grad on proj
        if av.proj.weight.grad is not None:
            stamp("av.proj.weight.grad", av.proj.weight.grad)
        # find first lora grad
        for n, p in av.named_parameters():
            if "lora_" in n and p.grad is not None:
                stamp(f"GRAD {n}", p.grad)
                break

    print("\n== AR: 1 backward step on real data ==")
    ar.zero_grad()
    with torch.amp.autocast(device_type=device.type, dtype=dtype):
        h_hat = ar(ar_batch["input_ids"], ar_batch["attention_mask"])
        ar_loss = ((ar_batch["h"] - h_hat) ** 2).sum(dim=-1).mean()
    stamp("AR loss", ar_loss)
    if torch.isfinite(ar_loss):
        ar_loss.backward()
        if ar.head.weight.grad is not None:
            stamp("ar.head.weight.grad", ar.head.weight.grad)
        for n, p in ar.named_parameters():
            if "lora_" in n and p.grad is not None:
                stamp(f"GRAD {n}", p.grad)
                break


if __name__ == "__main__":
    main()
