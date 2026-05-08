"""Deterministic train/eval row split for Phase 3."""

from __future__ import annotations

import json
import random
from pathlib import Path


def make_or_load_split(
    n_rows: int,
    *,
    eval_size: int = 500,
    seed: int = 42,
    path: str | Path = "data/splits.json",
) -> tuple[list[int], list[int]]:
    """Returns (train_indices, eval_indices). Persists to `path` and reloads
    on subsequent calls so AV and AR see the exact same split."""
    p = Path(path)
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("seed") == seed and d.get("n_rows") == n_rows and len(d.get("eval", [])) == eval_size:
            return d["train"], d["eval"]
    rng = random.Random(seed)
    indices = list(range(n_rows))
    rng.shuffle(indices)
    eval_idx = sorted(indices[:eval_size])
    train_idx = sorted(indices[eval_size:])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "seed": seed, "n_rows": n_rows, "eval_size": eval_size,
        "train": train_idx, "eval": eval_idx,
    }))
    return train_idx, eval_idx
