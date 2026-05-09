#!/usr/bin/env python3
"""Phase 3 entry point — SFT warm-start of AV + AR + joint FVE eval.

Usage:
    python scripts/03_warmstart.py --variant text
    python scripts/03_warmstart.py --variant lens --time-budget-min 180
    python scripts/03_warmstart.py --variant text --max-steps 50 --min-steps 0 \\
        --time-budget-min 5 --batch 2  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nla.qwen import auto_batch_sizes, load_qwen, DEFAULT_MODEL_ID  # noqa: E402
from nla.av import AV  # noqa: E402
from nla.ar import AR  # noqa: E402
from nla.data_phase3 import (  # noqa: E402
    AVDataset, ARDataset, av_collate, ar_collate, load_phase3_inputs,
)
from nla.splits import make_or_load_split  # noqa: E402
from nla.train_warmstart import load_final, save_final, train_av, train_ar  # noqa: E402
from nla.fve import joint_fve, variance_of_targets  # noqa: E402


def _resolve_device(s: str) -> torch.device:
    if s == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(s)


def run_variant(args, variant: str) -> dict:
    """Run AV + AR training for one variant. Returns result dict (logged to manifest)."""
    print(f"\n{'='*70}\n[variant={variant}] starting\n{'='*70}")

    # Load data.
    corpus, summaries, h_all = load_phase3_inputs(
        corpus_path=args.corpus,
        summaries_path=args.summaries,
        activations_path=args.activations,
        layer_key=args.layer + "_l2",  # h3 → h3_l2
    )
    print(f"loaded {len(corpus)} rows; h shape {h_all.shape}; layer {args.layer}_l2")

    # Cap n if requested (for smoke).
    if args.n is not None and args.n < len(corpus):
        corpus = corpus[: args.n]
        summaries = summaries[: args.n]
        h_all = h_all[: args.n]

    train_idx, eval_idx = make_or_load_split(
        n_rows=len(corpus), eval_size=min(args.eval_size, len(corpus) // 5),
        seed=args.seed, path=args.splits_path,
    )
    print(f"split: {len(train_idx)} train / {len(eval_idx)} eval")

    # Load base + tokenizer.
    device = _resolve_device(args.device)
    print(f"loading {args.model_id} (4bit={not args.no_bnb}) on {device}")
    base, tok, act_id = load_qwen(
        args.model_id,
        use_4bit=(not args.no_bnb),
        device_map="auto" if device.type == "cuda" else {"": device.type},
        dtype=torch.float16 if device.type in ("cuda", "mps") else torch.float32,
    )

    # Build AV; AR shares the same wrapped base (PEFT multi-adapter).
    av = AV(base, act_id, d_substrate=384, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    ar = AR(av.base, d_substrate=384, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    av.to(device); ar.to(device)
    print(f"AV trainable params: {av.trainable_parameter_count():,}")
    print(f"AR trainable params: {ar.trainable_parameter_count():,}")

    # Pick batch sizes based on currently free GPU memory (after model loaded).
    if args.batch == 0 or args.eval_batch == 0:
        auto_train, auto_eval, info = auto_batch_sizes()
        if args.batch == 0:
            args.batch = auto_train
        if args.eval_batch == 0:
            args.eval_batch = auto_eval
        print(f"auto batch sizing: device={info.get('device')} "
              f"free={info.get('free_gb')} GB / total={info.get('total_gb')} GB → "
              f"train_batch={args.batch}, eval_batch={args.eval_batch}")
    else:
        print(f"explicit batch sizes: train={args.batch}, eval={args.eval_batch}")

    # Datasets + loaders.
    av_train_ds = AVDataset(train_idx, summaries, h_all, tok, variant=variant)
    av_eval_ds = AVDataset(eval_idx, summaries, h_all, tok, variant=variant)
    ar_train_ds = ARDataset(train_idx, summaries, h_all, tok, variant=variant)
    ar_eval_ds = ARDataset(eval_idx, summaries, h_all, tok, variant=variant)
    pad = tok.pad_token_id
    av_train_loader = DataLoader(av_train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=partial(av_collate, pad_token_id=pad))
    av_eval_loader = DataLoader(av_eval_ds, batch_size=args.eval_batch, shuffle=False,
                                 collate_fn=partial(av_collate, pad_token_id=pad))
    ar_train_loader = DataLoader(ar_train_ds, batch_size=args.batch, shuffle=True,
                                  collate_fn=partial(ar_collate, pad_token_id=pad))
    ar_eval_loader = DataLoader(ar_eval_ds, batch_size=args.eval_batch, shuffle=False,
                                 collate_fn=partial(ar_collate, pad_token_id=pad))

    # FVE eval is autoregressive Qwen 3B decode → expensive. Use a small subset
    # for the in-training callback; full subset for the final headline number.
    fve_train_idx = eval_idx[: args.fve_eval_size]
    fve_final_idx = eval_idx[: args.final_fve_eval_size]
    av_fve_train_loader = DataLoader(
        AVDataset(fve_train_idx, summaries, h_all, tok, variant=variant),
        batch_size=args.eval_batch, shuffle=False,
        collate_fn=partial(av_collate, pad_token_id=pad),
    )
    av_fve_final_loader = DataLoader(
        AVDataset(fve_final_idx, summaries, h_all, tok, variant=variant),
        batch_size=args.eval_batch, shuffle=False,
        collate_fn=partial(av_collate, pad_token_id=pad),
    )
    print(f"FVE eval rows: callback={len(fve_train_idx)}, final={len(fve_final_idx)}")

    # FVE denominator on the held-out h.
    h_var = variance_of_targets(h_all[eval_idx])
    print(f"Var(h_l2) over eval: {h_var:.4f}")

    # Time budget split: half for AV, rest carries over to AR.
    total_budget_sec = args.time_budget_min * 60.0
    av_budget_sec = total_budget_sec * 0.5

    # Train AV (skip if a final.pt already exists in --ckpt-root).
    t0 = time.time()
    av_save = Path(args.ckpt_root) / f"av_{variant}"
    loaded = load_final(av, av_save, device=device) if args.skip_if_trained else None
    if loaded is not None:
        print(f"\n[variant={variant}] AV already trained → loaded {av_save}/final.pt "
              f"(step={loaded['step']}, stop_reason={loaded['stop_reason']}, "
              f"keys={loaded['n_loaded_keys']})")
        av_result = {"stop_reason": "loaded_from_disk",
                     "final_step": int(loaded.get("step") or 0),
                     "elapsed_sec": 0.0,
                     "history": loaded.get("history") or []}
    else:
        print(f"\n[variant={variant}] training AV (budget {av_budget_sec/60:.0f} min)")
        av_result = train_av(
            av, av_train_loader, av_eval_loader,
            max_steps=args.max_steps, min_steps=args.min_steps,
            eval_every=args.eval_every, ckpt_every=args.ckpt_every,
            grad_accum=args.grad_accum, grad_clip=1.0,
            lr_lora=args.lr_lora, lr_proj=args.lr_proj, weight_decay=0.01,
            warmup_steps=200, patience=args.patience, min_delta=args.min_delta,
            time_budget_sec=av_budget_sec,
            save_dir=av_save, device=device, log_every=50,
        )
    av_elapsed = time.time() - t0
    print(f"AV: stop_reason={av_result['stop_reason']}, steps={av_result['final_step']}, "
          f"elapsed={av_elapsed/60:.1f}min")

    ar_budget_sec = max(60.0, total_budget_sec - av_elapsed)

    # Build the FVE-callback for AR's early-stop signal — uses a small subset
    # (default 64 rows) so a single eval doesn't burn an hour of decode time.
    @torch.no_grad()
    def fve_at_step(step: int) -> float:
        av.eval()
        result = joint_fve(
            av, ar, av_fve_train_loader,
            h_var=h_var, tokenizer=tok,
            pad_token_id=pad, eos_token_id=tok.eos_token_id,
            device=device, max_new_tokens=args.max_new_tokens,
            return_samples=0,
        )
        av.train()
        return result["fve"]

    # Train AR (skip if a final.pt already exists).
    t1 = time.time()
    ar_save = Path(args.ckpt_root) / f"ar_{variant}"
    loaded_ar = load_final(ar, ar_save, device=device) if args.skip_if_trained else None
    if loaded_ar is not None:
        print(f"\n[variant={variant}] AR already trained → loaded {ar_save}/final.pt "
              f"(step={loaded_ar['step']}, stop_reason={loaded_ar['stop_reason']}, "
              f"keys={loaded_ar['n_loaded_keys']})")
        ar_result = {"stop_reason": "loaded_from_disk",
                     "final_step": int(loaded_ar.get("step") or 0),
                     "elapsed_sec": 0.0,
                     "history": loaded_ar.get("history") or []}
    else:
        print(f"\n[variant={variant}] training AR (budget {ar_budget_sec/60:.0f} min)")
        ar_result = train_ar(
            ar, ar_train_loader, ar_eval_loader,
            fve_eval_fn=fve_at_step,
            max_steps=args.max_steps, min_steps=args.min_steps,
            eval_every=args.eval_every, ckpt_every=args.ckpt_every,
            grad_accum=args.grad_accum, grad_clip=1.0,
            lr_lora=args.lr_lora, lr_proj=args.lr_proj, weight_decay=0.01,
            warmup_steps=200, patience=args.patience, min_delta=args.min_delta,
            time_budget_sec=ar_budget_sec,
            save_dir=ar_save, device=device, log_every=50,
        )
    ar_elapsed = time.time() - t1
    print(f"AR: stop_reason={ar_result['stop_reason']}, steps={ar_result['final_step']}, "
          f"elapsed={ar_elapsed/60:.1f}min")

    # Final FVE eval with sample dump for the report. Larger subset than the
    # callback so the headline number has reasonable variance.
    print(f"\n[variant={variant}] final joint FVE eval (n={len(fve_final_idx)})")
    av.eval(); ar.eval()
    final = joint_fve(
        av, ar, av_fve_final_loader,
        h_var=h_var, tokenizer=tok,
        pad_token_id=pad, eos_token_id=tok.eos_token_id,
        device=device, max_new_tokens=args.max_new_tokens,
        return_samples=10,
    )
    print(f"FVE = {final['fve']:.4f}  (MSE = {final['mse']:.4f}, n = {final['n']})")

    # Write history.
    Path(args.history_root).mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_root) / f"history_{variant}.json"
    history_path.write_text(json.dumps({
        "variant": variant,
        "av": av_result,
        "ar": ar_result,
        "final_fve": final["fve"],
        "final_mse": final["mse"],
        "h_var": h_var,
        "samples": final["samples"],
        "elapsed_total_sec": time.time() - t0,
    }, indent=2))
    print(f"wrote {history_path}")

    return {
        "variant": variant,
        "av_stop_reason": av_result["stop_reason"],
        "ar_stop_reason": ar_result["stop_reason"],
        "av_steps": av_result["final_step"],
        "ar_steps": ar_result["final_step"],
        "fve": final["fve"],
        "mse": final["mse"],
        "h_var": h_var,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 3: SFT warm-start AV + AR")
    p.add_argument("--variant", choices=["text", "lens", "both"], default="both",
                   help="text → summary_text; lens → summary_lens; both → run sequentially")
    p.add_argument("--layer", default="h3", help="activation layer key prefix (e.g. h3 → reads h3_l2)")
    p.add_argument("--n", type=int, default=None, help="cap on rows for smoke runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--min-steps", type=int, default=1000)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.005)
    p.add_argument("--time-budget-min", type=int, default=180,
                   help="hard wall-clock cap per variant (AV+AR combined)")
    p.add_argument("--batch", type=int, default=0,
                   help="train batch (0 = auto from free GPU memory)")
    p.add_argument("--eval-batch", type=int, default=0,
                   help="FVE eval batch (0 = auto from free GPU memory)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lr-proj", type=float, default=1e-3)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=60,
                   help="AV greedy max-new-tokens during FVE. Summaries average 50-65 toks.")
    p.add_argument("--fve-eval-size", type=int, default=64,
                   help="rows for the in-training FVE callback (called every --eval-every steps)")
    p.add_argument("--final-fve-eval-size", type=int, default=200,
                   help="rows for the final post-training FVE; the headline number")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--no-bnb", action="store_true", help="skip bitsandbytes 4-bit (fp16 fallback)")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--corpus", default="data/corpus.jsonl")
    p.add_argument("--summaries", default="data/summaries.jsonl")
    p.add_argument("--activations", default="data/activations.npz")
    p.add_argument("--splits-path", default="data/splits.json")
    p.add_argument("--ckpt-root", default="checkpoints/phase3")
    p.add_argument("--skip-if-trained", action=argparse.BooleanOptionalAction, default=True,
                   help="if --ckpt-root/<av|ar>_<variant>/final.pt exists, load it and skip training")
    p.add_argument("--history-root", default="data")
    p.add_argument("--manifest", default="data/MANIFEST_phase3.json")
    args = p.parse_args()

    variants = ["text", "lens"] if args.variant == "both" else [args.variant]

    results: list[dict] = []
    for variant in variants:
        res = run_variant(args, variant)
        results.append(res)
        # Free GPU memory between variants.
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Manifest.
    phase1, phase2 = {}, {}
    try:
        phase1 = json.loads(Path("data/MANIFEST.json").read_text())
    except Exception:
        pass
    try:
        phase2 = json.loads(Path("data/MANIFEST_phase2.json").read_text())
    except Exception:
        pass

    manifest = {
        "phase": 3,
        "model_id": args.model_id,
        "use_4bit": not args.no_bnb,
        "layer": args.layer,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lr_lora": args.lr_lora,
        "lr_proj": args.lr_proj,
        "batch": args.batch,
        "grad_accum": args.grad_accum,
        "max_steps": args.max_steps,
        "min_steps": args.min_steps,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "time_budget_min": args.time_budget_min,
        "results": results,
        "phase1_hf_revision": phase1.get("hf_revision"),
        "phase2_model": phase2.get("model"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.manifest}")

    # Decision banner.
    if len(results) == 2:
        ft = next(r["fve"] for r in results if r["variant"] == "text")
        fl = next(r["fve"] for r in results if r["variant"] == "lens")
        print(f"\n=== decision ===")
        print(f"  FVE_text = {ft:.4f}, FVE_lens = {fl:.4f}, |Δ| = {abs(ft-fl):.4f}")
        if max(ft, fl) < 0.10:
            print("  HARD-FAIL: both variants below 0.10 doc threshold.")
        elif abs(ft - fl) < 0.05:
            print("  WINNER: text (Occam — both within 0.05, prefer simpler prompt for Phase 4).")
        elif fl > ft + 0.05:
            print("  WINNER: lens — flag steganography for Phase 4 day-1 diagnostic.")
        elif ft > fl + 0.05:
            print("  WINNER: text — drop lens variant for Phase 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
