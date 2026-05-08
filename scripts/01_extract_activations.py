#!/usr/bin/env python3
"""Phase 1 entry point — extract per-layer activations from GuppyLM.

Usage:
    python scripts/01_extract_activations.py --n 5000

Defaults match the locked Phase 1 plan (seed=42, min_len=16, layer-agnostic).
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nla.extract import extract, iter_snippets, write_manifest, write_outputs  # noqa: E402
from nla.load import load_substrate  # noqa: E402


def _read_revision(path: Path) -> str | None:
    if path.exists():
        return path.read_text().strip()
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 1: GuppyLM activation extraction")
    p.add_argument("--n", type=int, default=5000, help="Number of snippets to keep.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-len", type=int, default=16)
    p.add_argument("--max-seq-len", type=int, default=128)
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--source-jsonl", default="data/train.jsonl")
    p.add_argument("--device", default="auto", help="cpu | cuda | mps | auto")
    p.add_argument("--out-corpus", default="data/corpus.jsonl")
    p.add_argument("--out-activations", default="data/activations.npz")
    p.add_argument("--out-manifest", default="data/MANIFEST.json")
    args = p.parse_args()

    print(f"loading substrate from {args.checkpoint} (device={args.device})")
    model, tokenizer, config = load_substrate(args.checkpoint, args.tokenizer, device=args.device)
    print(f"  params: {model.param_count():,}  config: {config}")

    print(f"reading {args.source_jsonl}")
    snippets = iter_snippets(args.source_jsonl)
    print(f"  available: {len(snippets):,} snippets")
    if len(snippets) < args.n:
        print(f"  warning: only {len(snippets)} < requested n={args.n}", file=sys.stderr)

    t0 = time.time()
    rows, H, H_l2 = extract(
        model,
        tokenizer,
        snippets,
        n=args.n,
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        min_len=args.min_len,
    )
    elapsed = time.time() - t0
    n_kept = len(rows)
    print(f"  kept {n_kept}/{args.n} in {elapsed:.1f}s ({n_kept/elapsed:.1f} snippets/s)")

    write_outputs(rows, H, H_l2, args.out_corpus, args.out_activations)
    print(f"wrote {args.out_corpus}")
    print(f"wrote {args.out_activations}")

    revision = _read_revision(Path("checkpoints/REVISION"))
    write_manifest(
        args.out_manifest,
        seed=args.seed,
        n_requested=args.n,
        n_kept=n_kept,
        source_jsonl=args.source_jsonl,
        hf_revision=revision,
        elapsed_sec=elapsed,
        H=H,
        extra={
            "min_len": args.min_len,
            "max_seq_len": args.max_seq_len,
            "checkpoint": args.checkpoint,
            "tokenizer": args.tokenizer,
        },
    )
    print(f"wrote {args.out_manifest}")

    # Quick eyeball: per-layer norm summary.
    import numpy as np
    print("\nper-layer raw-norm (mean ± std):")
    for l, arr in enumerate(H):
        n = np.linalg.norm(arr, axis=1)
        print(f"  h{l}: {n.mean():7.3f} ± {n.std():.3f}   (min={n.min():.3f} max={n.max():.3f})")

    nan_count = sum(int((~np.isfinite(arr)).sum()) for arr in H)
    print(f"\nNaN/Inf count across all raw arrays: {nan_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
