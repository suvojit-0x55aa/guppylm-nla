"""Joint AV → AR pipeline + FVE eval.

  h_l ── AV (greedy) ──► z (text)
                              │
                              ▼  re-tokenize via AR's chat template
                          AR ──► ĥ
  SE = ‖h_l − ĥ‖²
  FVE = 1 − mean(SE) / Var(h_l)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data_phase3 import _build_ar_prompt_ids


def decode_av_output(tokenizer, full_ids: torch.Tensor, prompt_len: int) -> str:
    """Strip the prompt prefix and decode the generated tokens; trim at first <|im_end|>."""
    new_ids = full_ids[prompt_len:].tolist()
    text = tokenizer.decode(new_ids, skip_special_tokens=False)
    for stop in ("<|im_end|>", tokenizer.eos_token):
        if stop and stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


@torch.no_grad()
def joint_fve(
    av,
    ar,
    av_eval_loader: DataLoader,
    *,
    h_var: float,
    tokenizer,
    pad_token_id: int,
    eos_token_id: int,
    device: torch.device,
    max_new_tokens: int = 80,
    return_samples: int = 0,
) -> dict:
    """Args:
      av_eval_loader: yields AVDataset batches (h, input_ids, attention_mask, ...)
      h_var: variance of h_l on the same eval set, computed once outside this fn
      return_samples: if > 0, also return up to this many (h, z, h_hat) traces
    """
    se_total, n = 0.0, 0
    samples = []
    for batch in tqdm(av_eval_loader, desc="FVE"):
        h = batch["h"].to(device)                                       # (B, 384)
        av_input_ids = batch["input_ids"].to(device)
        av_attn = batch["attention_mask"].to(device)
        # AV uses ONLY the prompt portion (labels[:prompt_len]==-100). Reconstruct
        # the prompt by selecting where labels is masked.
        prompt_mask = (batch["labels"] == -100) & (batch["attention_mask"] == 1)
        prompt_lens = prompt_mask.sum(dim=1)                            # (B,)
        # All rows share the same fixed AV prompt structure → use min for greedy start.
        # We still pass the full padded (input_ids, attention_mask) but only the prompt
        # contributes via attention_mask (rest is pad). For batched generation the
        # caller should ensure all rows have identical prompt_lens; AVDataset does.
        full_ids = av.generate(
            input_ids=av_input_ids[:, :prompt_lens.max()],
            attention_mask=av_attn[:, :prompt_lens.max()],
            h_l=h,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        # Decode each row's summary, build AR prompts.
        ar_input_ids = []
        decoded_summaries = []
        for b in range(full_ids.shape[0]):
            summary = decode_av_output(tokenizer, full_ids[b], int(prompt_lens[b].item()))
            decoded_summaries.append(summary)
            ar_ids = _build_ar_prompt_ids(tokenizer, summary)
            ar_input_ids.append(ar_ids)
        # Pad AR inputs.
        max_ar = max(len(x) for x in ar_input_ids)
        B = full_ids.shape[0]
        ar_pad = torch.full((B, max_ar), pad_token_id, dtype=torch.long, device=device)
        ar_attn = torch.zeros((B, max_ar), dtype=torch.long, device=device)
        for i, ids in enumerate(ar_input_ids):
            ar_pad[i, : len(ids)] = torch.tensor(ids, device=device)
            ar_attn[i, : len(ids)] = 1
        h_hat = ar(input_ids=ar_pad, attention_mask=ar_attn)            # (B, 384) fp32

        se = ((h - h_hat) ** 2).sum(dim=-1)                             # (B,)
        se_total += float(se.sum().item()); n += int(B)

        for b in range(B):
            if len(samples) >= return_samples:
                break
            samples.append({
                "h_norm": float(h[b].norm().item()),
                "h_hat_norm": float(h_hat[b].norm().item()),
                "se": float(se[b].item()),
                "summary": decoded_summaries[b],
            })

    mse = se_total / max(n, 1)
    fve = 1.0 - mse / max(h_var, 1e-8)
    return {"mse": mse, "fve": fve, "n": n, "h_var": h_var, "samples": samples}


def variance_of_targets(h: np.ndarray) -> float:
    """Var(h) over the eval-set rows. Used as FVE denominator."""
    return float(np.var(h, axis=0).sum())
