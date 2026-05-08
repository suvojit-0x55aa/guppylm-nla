# Phase 1 — Activation Extraction Report

**Date**: 2026-05-09 · **Substrate**: `arman-bd/guppylm-9M` @ `5f09d63fae6f5049bd2360d690f01b68a63d837b` · **N**: 5000 · **Seed**: 42 · **Throughput**: 89.3 snippets/s on M-series CPU.

## Outputs

| File | Size | Schema |
|------|------|--------|
| `data/corpus.jsonl` | 1.1 MB | `{row, L, ids, text, category}` per line |
| `data/activations.npz` | 81 MB | 12 arrays — `h0..h5` (raw) and `h0_l2..h5_l2` (L2-normalized), each `(5000, 384) float32` |
| `data/MANIFEST.json` | 2.2 KB | seed, n, HF SHA, timestamps, per-layer norm stats, 6×6 cosine matrix |

All 6 sanity tests green (`pytest tests/test_extract.py`): shape, finiteness, unit-L2, layer-distinctness (no hook aliasing), corpus alignment, logit-lens plausibility.

## Per-layer raw-norm growth

Pre-norm transformer; final `LayerNorm` happens after `blocks[5]`. Residual stream grows monotonically with depth, as expected:

| Layer | mean | std | min | max |
|-------|------|-----|-----|-----|
| h0 | 4.56 | 2.29 | 2.33 | 12.82 |
| h1 | 7.24 | 4.84 | 3.70 | 24.32 |
| h2 | 9.19 | 5.77 | 4.62 | 29.03 |
| h3 | 11.20 | 6.19 | 5.26 | 32.14 |
| h4 | 13.33 | 5.75 | 5.89 | 32.09 |
| h5 | 15.91 | 5.21 | 7.00 | 31.96 |

L2-normalized variants `h0_l2..h5_l2` are unit-norm to atol 1e-4 (verified across all 5000 rows).

## Pairwise layer cosine matrix (mean over 5000 rows, on L2-normalized residuals)

|     | h0   | h1   | h2   | h3   | h4   | h5   |
|-----|------|------|------|------|------|------|
| h0  | 1.00 | 0.83 | 0.70 | 0.59 | 0.49 | 0.39 |
| h1  | 0.83 | 1.00 | 0.90 | 0.79 | 0.67 | 0.55 |
| h2  | 0.70 | 0.90 | 1.00 | 0.92 | 0.81 | 0.69 |
| h3  | 0.59 | 0.79 | 0.92 | 1.00 | 0.92 | 0.81 |
| h4  | 0.49 | 0.67 | 0.81 | 0.92 | 1.00 | 0.92 |
| h5  | 0.39 | 0.55 | 0.69 | 0.81 | 0.92 | 1.00 |

Adjacent-layer cosines max out at 0.92 (h3↔h4, h4↔h5) — well under the 0.99 alias-detection threshold. Off-diagonal monotonically decays, confirming each block contributes a non-trivial residual delta. h0 vs h5 = 0.39 — substantial representational drift end-to-end.

## Logit-lens samples (sanity, 5 random rows × layers 3, 5)

```
row 2652 (L=20, cat=bubbles)
  text: 'user\nlook at those bubbles\nassistant\ni love bubbles. they go up and'
  h3 top-5: 'i'(0.99)  'new'(0.00)  'always'(0.00)  'not'(0.00)  'look'(0.00)
  h5 top-5: 'i'(1.00)  'my'(0.00)  'the'(0.00)  'always'(0.00)  'they'(0.00)

row 1235 (L=29, cat=night)
  text: 'user\nlights out time to rest\nassistant\nok sleep time. the dark is my quiet time. the oxygen sounds nice at'
  h3 top-5: 'night'(0.98)  'as'(0.00)  'out'(0.00)  'me'(0.00)  'of'(0.00)
  h5 top-5: 'night'(1.00)  'lights'(0.00)  'evening'(0.00)  'afternoon'(0.00)  'morning'(0.00)

row 3234 (L=27, cat=temp_cold)
  text: "user\nso cold\nassistant\nmy gills can tell it's colder. my belly move slower when it's cold."
  h3 top-5: ''(0.86)  'my'(0.05)  'i'(0.03)  'is'(0.02)  'can'(0.01)
  h5 top-5: ''(0.39)  'i'(0.34)  'my'(0.15)  'is'(0.11)  'can'(0.01)

row 395 (L=19, cat=bored)
  text: 'user\nis it boring in there\nassistant\ni could use some entertainment.'
  h3 top-5: 'or'(0.99)  'is'(0.00)  'entertainment'(0.00)  'but'(0.00)  'the'(0.00)
  h5 top-5: 'or'(1.00)  'in'(0.00)  'but'(0.00)  'much'(0.00)  'if'(0.00)

row 593 (L=17, cat=night)
  text: "user\nsweet dreams\nassistant\nnighty night. i'll be here"
  h3 top-5: '.'(0.99)  'will'(0.00)  'and'(0.00)  'when'(0.00)  'looks'(0.00)
  h5 top-5: '.'(1.00)  'and'(0.00)  'is'(0.00)  'will'(0.00)  'when'(0.00)
```

Every top-1 is a plausible next token. Notable: the model is **highly peaked** on the next token (often >0.99 at h3) — at this scale the substrate appears to commit early, with later layers refining rather than overturning.

## Observations to feed Phase 2

1. **Norm growth means the choice of raw vs L2-normalized matters more than at larger scale**. Raw h5 norms (mean 16, max 32) span 4.5× — AR loss `‖h - AR(z)‖²` will be dominated by high-norm samples unless we normalize. Plan stands: train AR against `h_l2` per the paper.
2. **Layer commitment is visible in cosine.** h3↔h4 = 0.92, h4↔h5 = 0.92 — the last two layers do less work than h2↔h3 (also 0.92) but the trajectory is uniform. The doc-spec primary layer (block 4 of 6 = `blocks[3]`) sits at the steepest semantic-change region.
3. **Logit-lens is sharp.** Tied weights make it cheap and Phase 2 prompts can include it as auxiliary signal: e.g. "the model is about to predict {top-3 tokens}; describe its full internal state".
4. **Final-token semantics work.** The truncations land on natural mid-sentence positions and the next-token prediction is coherent — Phase 2 teacher prompts can be straightforwardly seeded from `text` without extra cleanup.
5. **No NaN/Inf** anywhere. No `clamp` or hygiene needed downstream.

## Reproducing

```bash
bash scripts/00_fetch_substrate.sh
.venv/bin/python scripts/01_extract_activations.py --n 5000
pytest
```

Same seed → identical outputs (deterministic shuffle + truncation, `model.eval()` zeros dropout).
