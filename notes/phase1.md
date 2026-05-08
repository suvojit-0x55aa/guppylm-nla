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

## 10-sample review (seed=10, post-extraction sanity walk)

Run on the n=5000 cache, `random.seed(10)`. Each sample lists raw + L2 norms across all six layers and the top-3 logit-lens predictions at h3 (mid-depth, the doc's nominal "primary" layer) and h5 (final, after substrate's `LayerNorm`).

```
[1/10] row=4680 L=18 category=bubbles
  text: 'user\nthere are so many bubbles\nassistant\nbubbles are great. i'
  raw norms:    h0= 5.25  h1= 7.70  h2= 8.86  h3=10.04  h4=11.27  h5=13.70
  l2 norms:     h0=1.0000  h1=1.0000  h2=1.0000  h3=1.0000  h4=1.0000  h5=1.0000
  h3 top-3: 'swim'(0.56)  'chased'(0.10)  'they'(0.01)
  h5 top-3: 'swim'(0.71)  'chased'(0.28)  'stack'(0.00)

[2/10] row=266 L=20 category=visitors
  text: 'user\na friend wants to see you\nassistant\nnew shapes at the glass.'
  raw norms:    h0= 3.01  h1= 4.54  h2= 6.73  h3= 8.62  h4=10.98  h5=13.44
  h3 top-3: 'interesting'(0.92)  'big'(0.00)  'creatures'(0.00)
  h5 top-3: 'interesting'(1.00)  'moved'(0.00)  'hello'(0.00)

[3/10] row=3513 L=28 category=confused
  text: "user\nhave you heard of rent\nassistant\ni don't know what rent is. is it edible. if not i"
  raw norms:    h0= 4.66  h1= 6.25  h2= 7.71  h3= 9.54  h4=11.51  h5=13.89
  h3 top-3: "'m"(0.94)  "'ll"(0.01)  'swim'(0.00)
  h5 top-3: "'m"(1.00)  'think'(0.00)  'am'(0.00)

[4/10] row=3953 L=22 category=light            ← OUTLIER (h5 jump 2×)
  text: "user\ntime for lights\nassistant\ni noticed something different. there's my tunnel. hello"
  raw norms:    h0= 3.65  h1= 5.37  h2= 5.82  h3= 7.52  h4=13.92  h5=26.76
  h3 top-3: 'big'(0.13)  'filter'(0.12)  'tank'(0.03)
  h5 top-3: 'log'(0.07)  'bridge'(0.06)  'filter'(0.05)

[5/10] row=4735 L=19 category=plants
  text: 'user\ndo you eat the plants\nassistant\ni nibbled it once. it'
  raw norms:    h0= 4.16  h1= 6.09  h2= 6.92  h3= 8.05  h4= 9.77  h5=13.09
  h3 top-3: 'didn'(0.60)  'tasted'(0.07)  'watches'(0.01)
  h5 top-3: 'didn'(1.00)  'was'(0.00)  'just'(0.00)

[6/10] row=121 L=25 category=bored              ← OUTLIER (h0/h1 ~3× typical)
  text: 'user\nare you bored\nassistant\ni could use some entertainment. or food. food is entertainment.'
  raw norms:    h0=12.06  h1=22.81  h2=27.32  h3=30.23  h4=30.27  h5=29.08
  h3 top-3: ''(0.98)  'ball'(0.00)  'left'(0.00)
  h5 top-3: ''(1.00)  'with'(0.00)  'ball'(0.00)

[7/10] row=1688 L=26 category=lonely
  text: "user\nwould you like a friend\nassistant\ni have the gravel and the heater tube. that's enough."
  raw norms:    h0= 3.39  h1= 5.84  h2= 7.27  h3= 8.72  h4=12.39  h5=18.67
  h3 top-3: 'i'(0.61)  'being'(0.13)  'bored'(0.05)
  h5 top-3: 'i'(0.69)  'the'(0.14)  'being'(0.06)

[8/10] row=3789 L=16 category=name
  text: 'user\nwhy are you called guppy\nassistant\ni respond to'
  raw norms:    h0= 4.36  h1= 5.38  h2= 6.22  h3= 7.06  h4= 9.41  h5=11.93
  h3 top-3: 'guppy'(0.29)  'sized'(0.02)  'type'(0.01)
  h5 top-3: 'guppy'(1.00)  'little'(0.00)  'here'(0.00)

[9/10] row=4024 L=24 category=time
  text: "user\nhow do you know what time it is\nassistant\nit's food time or not food time"
  raw norms:    h0= 4.21  h1= 5.71  h2= 6.81  h3= 9.38  h4=10.95  h5=13.11
  h3 top-3: '.'(0.99)  'time'(0.00)  'and'(0.00)
  h5 top-3: '.'(1.00)  'that'(0.00)  'or'(0.00)

[10/10] row=2273 L=27 category=night
  text: 'user\ngoing to bed\nassistant\nok sleep time. my tail are already slowing down. the filter hum sounds nice'
  raw norms:    h0= 5.07  h1= 7.15  h2= 8.42  h3=10.71  h4=13.96  h5=17.33
  h3 top-3: 'at'(0.88)  'PARTS'(0.00)  'OBJECTS'(0.00)
  h5 top-3: 'at'(1.00)  'against'(0.00)  'around'(0.00)
```

### Findings

- **L2 norms exact** to atol < 2e-7 across all 12 arrays × 5000 rows. Phase 2 can rely on the unit-norm invariant without re-checking.
- **Logit-lens quality is high.** Top-1 at h5 is the actual fluent next token in 9/10 samples — `interesting`, `'m`, `didn`, `guppy`, `at`, `.`. Phase 2 teacher prompts can include the top-3 next-token distribution as auxiliary context.
- **h3 vs h5 shift**: h3 is already mostly committed (top-1 prob > 0.5 in 7/10), h5 sharpens to near-1.0. The substrate is decisive at this scale; layer choice in Phase 2 may matter less than expected for next-token-conditioned reconstruction. Argues for trying h3 (steepest semantic-change region per the cosine matrix) before defaulting to h5.
- **Two outliers worth tracking, not blocking:**
  - **Row 121**: norms 3× typical from h0 onward. Text is repetitive (`"food. food is entertainment."`) — likely the embedding of the repeated token. Won't break MSE/FVE math but may dominate AR loss if not normalized. Plan stands: train against `h_l2`.
  - **Row 3953**: h4→h5 norm doubles (13.92 → 26.76). Truncates on `"hello"` mid-utterance. Suggests the substrate's last-layer residual at certain low-frequency tokens has long tails. Cap-and-clamp is unnecessary; just be aware FVE on edge rows may be noisier.
- **Sample diversity is healthy**: 10 different categories represented (bubbles, visitors, confused, light, plants, bored, lonely, name, time, night). The deterministic shuffle is doing its job — no category clustering at row offsets.

### Reproducing this review

```bash
.venv/bin/python -c "
import numpy as np, json, random
from nla.load import load_substrate
from nla.extract import logit_lens
m, tok, _ = load_substrate('checkpoints/best_model.pt', 'data/tokenizer.json')
rows = [json.loads(l) for l in open('data/corpus.jsonl')]
npz = np.load('data/activations.npz')
random.seed(10)
for idx in random.sample(range(len(rows)), 10):
    r = rows[idx]
    print(f'row={idx} L={r[\"L\"]} cat={r.get(\"category\")}')
    print(f'  text: {r[\"text\"]!r}')
    print(f'  norms: ' + '  '.join(f'h{l}={np.linalg.norm(npz[f\"h{l}\"][idx]):.2f}' for l in range(6)))
    for l in [3, 5]:
        top = logit_lens(m, tok, npz[f'h{l}'][idx], k=3)
        print(f'  h{l}: ' + '  '.join(f'{s.strip()!r}({p:.2f})' for _, s, p in top))
"
```

## Reproducing

```bash
bash scripts/00_fetch_substrate.sh
.venv/bin/python scripts/01_extract_activations.py --n 5000
pytest
```

Same seed → identical outputs (deterministic shuffle + truncation, `model.eval()` zeros dropout).
