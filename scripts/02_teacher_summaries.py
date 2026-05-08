#!/usr/bin/env python3
"""Phase 2 entry point — teacher summaries for each Phase-1 row.

Reads:  data/corpus.jsonl, data/activations.npz, checkpoints/* (for logit-lens)
Writes: data/summaries.jsonl (row-aligned), data/MANIFEST_phase2.json

Usage:
    export OPENAI_API_KEY=sk-...
    python scripts/02_teacher_summaries.py --n 10                # smoke
    python scripts/02_teacher_summaries.py --n 5000 --yes        # full
    python scripts/02_teacher_summaries.py --n 1 --dry-run       # eyeball prompts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.asyncio import tqdm as atqdm

# Allow running as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nla.extract import iter_snippets, logit_lens  # noqa: E402
from nla.load import load_substrate  # noqa: E402
from nla.teacher import (  # noqa: E402
    SYSTEM_PROMPT,
    OpenAITeacher,
    already_done,
    build_user_text_only,
    build_user_with_lens,
    is_refusal,
    summarize_row,
)


# Pricing (USD per 1M tokens) — gpt-4o-mini as of 2026-Q1. Update if needed.
PRICING = {
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
    "gpt-5-mini":  {"in": 0.25, "out": 2.00},
    "gpt-4o":      {"in": 5.00, "out": 15.00},
}


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(model, PRICING["gpt-4o-mini"])
    return (in_tok / 1e6) * p["in"] + (out_tok / 1e6) * p["out"]


# ── Logit-lens precomputation ─────────────────────────────────────────────────


def compute_top3(model, tokenizer, h5: np.ndarray, k: int = 3) -> list[list[tuple[str, float]]]:
    """Run logit_lens for every row of h5. Returns list of [(tok_str, prob), ...]."""
    out: list[list[tuple[str, float]]] = []
    for i in range(h5.shape[0]):
        top = logit_lens(model, tokenizer, h5[i], k=k)
        out.append([(tok, prob) for _id, tok, prob in top])
    return out


# ── Resume / output ───────────────────────────────────────────────────────────


def load_existing(out_path: Path) -> dict[int, dict]:
    if not out_path.exists():
        return {}
    rows: dict[int, dict] = {}
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rows[int(r["row"])] = r
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return rows


def write_jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


# ── Async driver ──────────────────────────────────────────────────────────────


async def run_async(
    *,
    teacher,
    rows: list[dict],
    top3_all: list[list[tuple[str, float]]],
    todo_indices: list[int],
    out_path: Path,
    concurrency: int,
    retry_strict: bool,
    max_tokens: int,
    fsync_every: int = 100,
) -> tuple[int, int, int, int]:
    """Returns (n_completed, n_errors, total_in_tokens, total_out_tokens)."""
    sem = asyncio.Semaphore(concurrency)
    n_errors = 0
    in_tok_total = 0
    out_tok_total = 0
    completed = 0
    fsync_lock = asyncio.Lock()
    file_handle = open(out_path, "a", buffering=1)

    async def worker(i: int):
        nonlocal n_errors, in_tok_total, out_tok_total, completed
        async with sem:
            r = rows[i]
            text = r["text"]
            top3 = top3_all[i]
            try:
                result = await summarize_row(
                    teacher, text, top3, retry_strict=retry_strict, max_tokens=max_tokens
                )
            except Exception as e:  # belt-and-suspenders; OpenAITeacher already retries
                result = {"model": teacher.model, "error": {"all": f"unexpected: {e!r}"}}
            result["row"] = i
            in_tok_total += int(result.get("input_tokens_text", 0)) + int(result.get("input_tokens_lens", 0))
            out_tok_total += int(result.get("output_tokens_text", 0)) + int(result.get("output_tokens_lens", 0))
            if "error" in result:
                n_errors += 1

            async with fsync_lock:
                file_handle.write(json.dumps(result) + "\n")
                completed += 1
                if completed % fsync_every == 0:
                    file_handle.flush()
                    os.fsync(file_handle.fileno())

    try:
        await atqdm.gather(*[worker(i) for i in todo_indices], desc="teacher")
    finally:
        file_handle.flush()
        os.fsync(file_handle.fileno())
        file_handle.close()
    return completed, n_errors, in_tok_total, out_tok_total


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2: teacher summaries for each row.")
    p.add_argument("--n", type=int, default=None, help="rows to process (default: all)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--teacher", default="openai:gpt-4o-mini",
                   help="format: openai:<model> | (future) ollama:<model>")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cost-cap-usd", type=float, default=5.0)
    p.add_argument("--yes", action="store_true", help="skip cost-cap confirmation")
    p.add_argument("--include-lens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--retry-strict", action="store_true", help="re-issue refusals with stricter prompt")
    p.add_argument("--corpus", default="data/corpus.jsonl")
    p.add_argument("--activations", default="data/activations.npz")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--tokenizer", default="data/tokenizer.json")
    p.add_argument("--out", default="data/summaries.jsonl")
    p.add_argument("--manifest", default="data/MANIFEST_phase2.json")
    p.add_argument("--dry-run", action="store_true", help="print prompts for row 0 and exit")
    p.add_argument("--no-resume", action="store_true", help="ignore existing output, start fresh")
    args = p.parse_args()

    # Parse teacher spec.
    if not args.teacher.startswith("openai:"):
        print(f"error: only openai:* teachers are wired up. got: {args.teacher}", file=sys.stderr)
        return 2
    model_name = args.teacher.split(":", 1)[1]

    # Load corpus + activations.
    rows = iter_snippets(args.corpus)
    npz = np.load(args.activations)
    h5 = npz["h5"]
    if h5.shape[0] != len(rows):
        print(f"error: corpus rows ({len(rows)}) != activations rows ({h5.shape[0]})", file=sys.stderr)
        return 2
    n = args.n if args.n is not None else len(rows)
    n = min(n, len(rows))
    rows = rows[:n]
    h5 = h5[:n]
    print(f"loaded {n} rows from {args.corpus}; activations shape {h5.shape}")

    # Load substrate, compute logit-lens top-3 for every row.
    print(f"loading substrate from {args.checkpoint}")
    model, tokenizer, _cfg = load_substrate(args.checkpoint, args.tokenizer, device="cpu")
    print("computing logit-lens top-3 per row")
    t0 = time.time()
    top3_all = compute_top3(model, tokenizer, h5, k=3)
    print(f"  done in {time.time()-t0:.1f}s")

    if args.dry_run:
        print("\n=== DRY-RUN: prompts for row 0 ===\n")
        print("--- system ---")
        print(SYSTEM_PROMPT)
        print("\n--- user (text-only) ---")
        print(build_user_text_only(rows[0]["text"]))
        print("\n--- user (with logit-lens) ---")
        print(build_user_with_lens(rows[0]["text"], top3_all[0]))
        print("\n=== END DRY-RUN ===")
        return 0

    # Resume.
    out_path = Path(args.out)
    existing = {} if args.no_resume else load_existing(out_path)
    todo = [i for i in range(n) if not already_done(existing.get(i), include_lens=args.include_lens)]
    print(f"resume: {len(existing)} existing rows, {n - len(todo)} skipped, {len(todo)} to do")
    if not todo:
        print("nothing to do.")
        return 0

    # Teacher.
    teacher = OpenAITeacher(model=model_name, seed=args.seed)
    print(f"teacher: openai:{model_name} (T={args.temperature}, max_tokens={args.max_tokens}, "
          f"concurrency={args.concurrency}, retry_strict={args.retry_strict})")

    # Pre-flight: cost extrapolation from one paired call on the first todo row.
    print("pre-flight: 1 paired call to estimate cost")
    sample_idx = todo[0]
    sample_result = asyncio.run(summarize_row(
        teacher, rows[sample_idx]["text"], top3_all[sample_idx],
        retry_strict=args.retry_strict, max_tokens=args.max_tokens,
    ))
    sample_in = int(sample_result.get("input_tokens_text", 0)) + int(sample_result.get("input_tokens_lens", 0))
    sample_out = int(sample_result.get("output_tokens_text", 0)) + int(sample_result.get("output_tokens_lens", 0))
    sample_cost = estimate_cost(model_name, sample_in, sample_out)
    estimated_total = sample_cost * len(todo)
    print(f"  sample row: in={sample_in} out={sample_out} cost=${sample_cost:.4f}")
    print(f"  estimated total for {len(todo)} rows: ${estimated_total:.2f}")
    if estimated_total > args.cost_cap_usd and not args.yes:
        print(f"\nERROR: estimated ${estimated_total:.2f} exceeds --cost-cap-usd ${args.cost_cap_usd:.2f}.\n"
              "Pass --yes to proceed.", file=sys.stderr)
        return 3

    # Persist the pre-flight sample so we don't pay twice for it.
    sample_result["row"] = sample_idx
    write_jsonl_append(out_path, sample_result)
    todo = todo[1:]

    # Main async loop.
    print(f"starting {len(todo)} async calls, concurrency={args.concurrency}")
    t1 = time.time()
    completed, n_errors, in_tok_total, out_tok_total = asyncio.run(run_async(
        teacher=teacher, rows=rows, top3_all=top3_all, todo_indices=todo,
        out_path=out_path, concurrency=args.concurrency,
        retry_strict=args.retry_strict, max_tokens=args.max_tokens,
    ))
    elapsed = time.time() - t1
    # Add the pre-flight sample's tokens into the total.
    in_tok_total += sample_in
    out_tok_total += sample_out
    completed += 1
    if sample_result.get("error"):
        n_errors += 1

    total_cost = estimate_cost(model_name, in_tok_total, out_tok_total)
    print(f"\ncompleted {completed} rows in {elapsed:.1f}s ({completed/max(elapsed,1):.1f} rows/s)")
    print(f"errors: {n_errors}/{completed}  ({100*n_errors/max(completed,1):.2f}%)")
    print(f"tokens: in={in_tok_total:,}  out={out_tok_total:,}  cost=${total_cost:.4f}")

    # Quality-gate report on full output (including resumed rows).
    all_rows = [json.loads(l) for l in open(out_path)]
    summary = {"text": [], "lens": []}
    refusal_in_success = {"text": 0, "lens": 0}
    for r in all_rows:
        for var in ("text", "lens"):
            key = f"summary_{var}"
            if key in r:
                summary[var].append(len(r[key].split()))
                if is_refusal(r[key]):
                    refusal_in_success[var] += 1

    def _quantiles(xs: list[int]) -> tuple[int, int, int]:
        if not xs:
            return (0, 0, 0)
        xs = sorted(xs)
        return xs[len(xs)//10], xs[len(xs)//2], xs[len(xs)*9//10]

    p10t, p50t, p90t = _quantiles(summary["text"])
    p10l, p50l, p90l = _quantiles(summary["lens"])
    error_count = sum(1 for r in all_rows if "error" in r)
    refusal_rate = (refusal_in_success["text"] + refusal_in_success["lens"]) / max(2 * len(all_rows), 1)

    print(f"\n=== quality-gate report ({len(all_rows)} total rows) ===")
    print(f"  errors: {error_count}/{len(all_rows)}  ({100*error_count/max(len(all_rows),1):.3f}%)")
    print(f"  summary_text words: P10={p10t} P50={p50t} P90={p90t}  refusals_in_success={refusal_in_success['text']}")
    print(f"  summary_lens words: P10={p10l} P50={p50l} P90={p90l}  refusals_in_success={refusal_in_success['lens']}")
    if len(all_rows) <= 100 and refusal_rate > 0.001:
        print(f"\n[!] smoke refusal rate {refusal_rate*100:.2f}% > 0.1% — recommend --retry-strict for full run")

    # Manifest.
    phase1 = {}
    p1_path = Path("data/MANIFEST.json")
    if p1_path.exists():
        try:
            phase1 = json.load(open(p1_path))
        except Exception:
            pass
    manifest = {
        "phase": 2,
        "model": model_name,
        "system_prompt_persona_neutral": True,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "concurrency": args.concurrency,
        "retry_strict": args.retry_strict,
        "include_lens": args.include_lens,
        "n_total_rows": len(all_rows),
        "n_completed_this_run": completed,
        "n_errors": error_count,
        "elapsed_sec": round(elapsed, 2),
        "tokens_in": in_tok_total,
        "tokens_out": out_tok_total,
        "estimated_cost_usd": round(total_cost, 4),
        "summary_text_word_quantiles": {"p10": p10t, "p50": p50t, "p90": p90t},
        "summary_lens_word_quantiles": {"p10": p10l, "p50": p50l, "p90": p90l},
        "refusals_in_success": refusal_in_success,
        "phase1_hf_revision": phase1.get("hf_revision"),
        "phase1_seed": phase1.get("seed"),
        "phase1_n_kept": phase1.get("n_kept"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
