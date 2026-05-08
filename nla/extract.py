"""Phase 1 — activation extraction loop.

For each truncated snippet, run the substrate, take the post-residual
output of every Block at the final token, write text+ids to JSONL and
the per-layer (raw + L2-normalized) activations to NPZ. Row index aligns
across files.
"""

import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer
from tqdm import tqdm

from ._substrate import GuppyLM
from .hooks import register_block_hooks


def iter_snippets(jsonl_path: str | Path) -> List[dict]:
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract(
    model: GuppyLM,
    tokenizer: Tokenizer,
    snippets: List[dict],
    n: int,
    seed: int = 42,
    max_seq_len: int = 128,
    min_len: int = 16,
    device: torch.device | None = None,
) -> Tuple[List[dict], List[np.ndarray], List[np.ndarray]]:
    """Returns (rows, H, H_l2) where:
       rows[i] = {row, L, ids, text}
       H[l] = (n_kept, d_model) float32 — raw post-block residual at final token
       H_l2[l] = same, L2-normalized per row.
    """
    if device is None:
        device = next(model.parameters()).device
    rng = random.Random(seed)
    pool = list(snippets)
    rng.shuffle(pool)

    n_blocks = len(model.blocks)
    d_model = model.config.d_model
    H = [np.empty((n, d_model), dtype=np.float32) for _ in range(n_blocks)]
    rows: List[dict] = []

    storage, handles = register_block_hooks(model)
    kept = 0
    try:
        with torch.no_grad():
            for s in tqdm(pool, total=min(n, len(pool)), desc="extract"):
                if kept >= n:
                    break
                ids = tokenizer.encode(s["text"]).ids
                if len(ids) < min_len:
                    continue
                upper = min(len(ids), max_seq_len)
                if upper < min_len:
                    continue
                L = rng.randint(min_len, upper)
                truncated = ids[:L]
                for v in storage.values():
                    v.clear()
                x = torch.tensor([truncated], dtype=torch.long, device=device)
                model(x)
                for l in range(n_blocks):
                    out = storage[l][0]  # (1, L, d_model)
                    H[l][kept] = out[0, -1, :].to("cpu", dtype=torch.float32).numpy()
                rows.append({
                    "row": kept,
                    "L": L,
                    "ids": truncated,
                    "text": tokenizer.decode(truncated),
                    "category": s.get("category"),
                })
                kept += 1
    finally:
        for h in handles:
            h.remove()

    H = [arr[:kept] for arr in H]
    H_l2 = [
        arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-8)
        for arr in H
    ]
    return rows, H, H_l2


def write_outputs(
    rows: List[dict],
    H: List[np.ndarray],
    H_l2: List[np.ndarray],
    out_corpus: str | Path,
    out_activations: str | Path,
) -> None:
    Path(out_corpus).parent.mkdir(parents=True, exist_ok=True)
    with open(out_corpus, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    arrays = {f"h{l}": H[l] for l in range(len(H))}
    arrays.update({f"h{l}_l2": H_l2[l] for l in range(len(H_l2))})
    np.savez_compressed(str(out_activations), **arrays)


def diagnostics(H: List[np.ndarray]) -> dict:
    """Per-layer norm stats + pairwise layer cosine matrix on raw activations."""
    norms = [np.linalg.norm(arr, axis=1) for arr in H]
    per_layer = [
        {
            "mean": float(n.mean()),
            "std": float(n.std()),
            "min": float(n.min()),
            "max": float(n.max()),
        }
        for n in norms
    ]
    L = len(H)
    cosine = np.zeros((L, L), dtype=np.float32)
    eps = 1e-8
    for i in range(L):
        ui = H[i] / np.maximum(np.linalg.norm(H[i], axis=1, keepdims=True), eps)
        for j in range(L):
            uj = H[j] / np.maximum(np.linalg.norm(H[j], axis=1, keepdims=True), eps)
            cosine[i, j] = float((ui * uj).sum(axis=1).mean())
    return {"per_layer_norms": per_layer, "pairwise_cosine": cosine.tolist()}


def write_manifest(
    out_path: str | Path,
    *,
    seed: int,
    n_requested: int,
    n_kept: int,
    source_jsonl: str,
    hf_revision: str | None,
    elapsed_sec: float,
    H: List[np.ndarray],
    extra: dict | None = None,
) -> None:
    diag = diagnostics(H)
    manifest = {
        "phase": 1,
        "seed": seed,
        "n_requested": n_requested,
        "n_kept": n_kept,
        "source_jsonl": str(source_jsonl),
        "hf_revision": hf_revision,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed_sec, 2),
        "throughput_per_sec": round(n_kept / elapsed_sec, 2) if elapsed_sec > 0 else None,
        "n_layers": len(H),
        "d_model": int(H[0].shape[1]) if H else None,
        "per_layer_norms": diag["per_layer_norms"],
        "pairwise_cosine": diag["pairwise_cosine"],
    }
    if extra:
        manifest.update(extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)


def inspect(corpus_path: str | Path, npz_path: str | Path, n: int = 3, seed: int = 0) -> None:
    """Eyeball helper: print a few rows + their logit-lens top-5 (no model load)."""
    rng = random.Random(seed)
    rows = iter_snippets(corpus_path)
    npz = np.load(str(npz_path))
    n_layers = sum(1 for k in npz.files if k.startswith("h") and "_" not in k)
    print(f"corpus rows: {len(rows)}, npz arrays: {npz.files}")
    sample = rng.sample(range(len(rows)), min(n, len(rows)))
    for idx in sample:
        r = rows[idx]
        print(f"\n--- row {idx} (L={r['L']}, cat={r.get('category')}) ---")
        print(f"text: {r['text']!r}")
        for l in range(n_layers):
            v = npz[f"h{l}"][idx]
            v_l2 = npz[f"h{l}_l2"][idx]
            print(f"  h{l}: |raw|={np.linalg.norm(v):.3f}  |l2|={np.linalg.norm(v_l2):.5f}  v[:4]={v[:4]}")


def logit_lens(model: GuppyLM, tokenizer: Tokenizer, h_row: np.ndarray, k: int = 5) -> List[Tuple[int, str, float]]:
    """Project a single residual through final norm + tied lm_head, return top-k tokens."""
    with torch.no_grad():
        h = torch.tensor(h_row, dtype=torch.float32, device=next(model.parameters()).device).unsqueeze(0)
        logits = model.lm_head(model.norm(h))[0]
        probs = torch.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, k)
    out = []
    for v, i in zip(topv.tolist(), topi.tolist()):
        try:
            tok = tokenizer.decode([i])
        except Exception:
            tok = f"<id={i}>"
        out.append((i, tok, float(v)))
    return out
