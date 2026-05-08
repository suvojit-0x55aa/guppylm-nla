# guppylm-nla — Phase 1: activation extraction

POC of Anthropic's Natural Language Autoencoder (NLA) recipe with GuppyLM (9M, frozen) as the substrate. This repo currently implements **Phase 1** only — the activation cache that downstream phases (warm-start SFT, GRPO RL, FVE eval) consume.

Spec: `chief-of-staff/ideas/apply-activation-translation-to-guppy-lm.md`.
Plan: `~/.claude/plans/look-at-the-apply-activation-translation-structured-pebble.md`.

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Fetch substrate (one-shot, ~50MB)
bash scripts/00_fetch_substrate.sh

# Smoke (n=100, ~5s on CPU)
python scripts/01_extract_activations.py --n 100 \
    --out-corpus data/corpus_smoke.jsonl --out-activations data/activations_smoke.npz

# Tests
pytest

# Full extraction (n=5000, ~1-2 min)
python scripts/01_extract_activations.py --n 5000
```

## Outputs (data/)

| File | Schema |
|------|--------|
| `corpus.jsonl` | `{row, L, ids, text}` per line; `L` is truncation length, `ids` is `int[L]` token ids, `text` is the decoded prefix. |
| `activations.npz` | 12 arrays, all `(N, 384) float32`: `h0..h5` raw post-block residuals at the final token; `h0_l2..h5_l2` L2-normalized variants. |
| `MANIFEST.json` | seed, n, source path, HF SHA, timestamp, throughput, per-layer norm stats, 6×6 pairwise layer-cosine matrix. |

Row index aligns across all files: `corpus.jsonl[row] ↔ activations[*][row]`.

## Notes

- All 6 transformer blocks are hooked; the "primary" layer for Phase 2 is deferred (caching all is cheap).
- Truncation: `L ∈ [16, min(len(ids), 128)]`, deterministic per `seed=42`.
- Substrate is `arman-bd/guppylm-9M` from HuggingFace, SHA-pinned in `checkpoints/REVISION`.
