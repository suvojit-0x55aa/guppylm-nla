"""Datasets + collates for Phase 3 SFT training of AV and AR.

Reads:
  - data/corpus.jsonl (Phase 1)
  - data/summaries.jsonl (Phase 2)
  - data/activations.npz (Phase 1) — h3_l2 only

Yields per row:
  AVDataset → (h3_l2, prompt_ids, target_ids, full_ids, labels)
  ARDataset → (prompt_ids, attention_mask, h3_l2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


SYSTEM_AV = (
    "You are a verbalizer for a tiny chatbot's internal state. Given the model's "
    "internal activation (provided as a special token <ACT>), describe in one or "
    "two sentences (≤ 50 words) what the model is processing or about to produce. "
    "Be specific. No preamble, no quotes, just the description as plain prose."
)

SYSTEM_AR = (
    "You will read a description of a tiny chatbot's internal state. Your final "
    "hidden state will be used to reconstruct that internal activation as a vector. "
    "Internalize the description fully."
)


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_phase3_inputs(
    corpus_path: str | Path = "data/corpus.jsonl",
    summaries_path: str | Path = "data/summaries.jsonl",
    activations_path: str | Path = "data/activations.npz",
    *,
    layer_key: str = "h3_l2",
) -> tuple[list[dict], list[dict], np.ndarray]:
    corpus = _read_jsonl(corpus_path)
    summaries_by_row = {int(r["row"]): r for r in _read_jsonl(summaries_path)}
    summaries = [summaries_by_row.get(i) for i in range(len(corpus))]
    missing = [i for i, s in enumerate(summaries) if s is None]
    if missing:
        raise RuntimeError(f"missing summaries for rows: {missing[:10]}...")
    npz = np.load(activations_path)
    h = npz[layer_key]                                      # (N, 384) float32
    return corpus, summaries, h


N_ACT_TOKENS = 8                                                # injection-position count


def _build_av_prompt_ids(tokenizer, act_token: str = "<ACT>",
                          n_act: int = N_ACT_TOKENS) -> list[int]:
    """Tokenize the chat-template prompt for AV (system + user=<ACT>×N) with
    the assistant generation tag appended. Returns prompt-only ids — caller
    appends the target tokens.

    Multiple <ACT> tokens carry the SAME projected activation vector (via
    AV._inject's mask-broadcast). The repetition gives the activation signal
    more 'voice' through Qwen's self-attention stack — a single embedding
    position is too easy for the deep model to ignore."""
    messages = [
        {"role": "system", "content": SYSTEM_AV},
        {"role": "user", "content": act_token * n_act},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _build_ar_prompt_ids(tokenizer, summary: str) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_AR},
        {"role": "user", "content": summary},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _normalize_variant_key(variant: str) -> str:
    """Accept short ('text', 'lens') or full ('summary_text', 'summary_lens')."""
    if variant in ("text", "lens"):
        return f"summary_{variant}"
    return variant


class AVDataset(Dataset):
    def __init__(
        self,
        rows_indices: Sequence[int],
        summaries: list[dict],
        h: np.ndarray,
        tokenizer,
        *,
        variant: str = "summary_text",
        max_target_tokens: int = 80,
    ):
        self.rows_indices = list(rows_indices)
        self.summaries = summaries
        self.h = h
        self.tok = tokenizer
        self.variant = _normalize_variant_key(variant)
        self.max_target_tokens = max_target_tokens
        self.prompt_ids = _build_av_prompt_ids(tokenizer)               # constant per row
        self.eos_id = tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.rows_indices)

    def __getitem__(self, i: int) -> dict:
        row_id = self.rows_indices[i]
        summary = self.summaries[row_id][self.variant]
        target_ids = self.tok(summary, add_special_tokens=False)["input_ids"][: self.max_target_tokens]
        target_ids = target_ids + [self.eos_id]
        full_ids = self.prompt_ids + target_ids
        labels = [-100] * len(self.prompt_ids) + target_ids
        return {
            "h": torch.from_numpy(self.h[row_id].astype(np.float32)),
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def av_collate(batch: list[dict], pad_token_id: int) -> dict:
    """Left-pad: required by Qwen2 flash-attention-2 (it rejects right-pad).
    Pads go at the start; real tokens (prompt + target + eos) at the end.
    Labels are -100 on pad slots and on prompt slots; CE loss only sees the
    target tokens, regardless of padding side."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((B, max_len), -100, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    h = torch.stack([b["h"] for b in batch])                            # (B, 384)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, max_len - n:] = b["input_ids"]
        labels[i, max_len - n:] = b["labels"]
        attn[i, max_len - n:] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels, "h": h}


class ARDataset(Dataset):
    def __init__(
        self,
        rows_indices: Sequence[int],
        summaries: list[dict],
        h: np.ndarray,
        tokenizer,
        *,
        variant: str = "summary_text",
        max_prompt_tokens: int = 256,
    ):
        self.rows_indices = list(rows_indices)
        self.summaries = summaries
        self.h = h
        self.tok = tokenizer
        self.variant = _normalize_variant_key(variant)
        self.max_prompt_tokens = max_prompt_tokens

    def __len__(self) -> int:
        return len(self.rows_indices)

    def __getitem__(self, i: int) -> dict:
        row_id = self.rows_indices[i]
        summary = self.summaries[row_id][self.variant]
        prompt_ids = _build_ar_prompt_ids(self.tok, summary)[: self.max_prompt_tokens]
        return {
            "h": torch.from_numpy(self.h[row_id].astype(np.float32)),
            "input_ids": torch.tensor(prompt_ids, dtype=torch.long),
        }


def ar_collate(batch: list[dict], pad_token_id: int) -> dict:
    """Left-pad: required by Qwen2 flash-attention-2. Real tokens at the end;
    AR's last-token gather can simply use the final index."""
    max_len = max(b["input_ids"].shape[0] for b in batch)
    B = len(batch)
    input_ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((B, max_len), dtype=torch.long)
    h = torch.stack([b["h"] for b in batch])
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids[i, max_len - n:] = b["input_ids"]
        attn[i, max_len - n:] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "h": h}
