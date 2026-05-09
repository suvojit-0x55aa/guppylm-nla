# Phase 3 — Apply Activation Translation to GuppyLM (NEGATIVE RESULT)

**Status: HALTED.** Phase 3 SFT warm-start of AV+AR is infeasible at this scale on this substrate. The substrate carries `h_l → category` signal (linear-probe lift +0.97) but the proposed AV architecture (`<ACT>` injection + LoRA-r16 on q/v projections of frozen 4-bit Qwen2.5-3B) cannot extract that signal into free-text summaries. Held-out cross-entropy plateaus at the prompt prior regardless of injection variant.

This document records every iteration, the empirical floor, the sample data the verdict is based on, and the open questions for any architecture-redesign successor.

---

## 1. Plan recap

`look-at-the-apply-activation-translation-structured-pebble.md` (in `~/.claude/plans/`) prescribed:

- Substrate: 9M-param GuppyLM. h3_l2 = L2-normalised residual at block 4/6 ≈ 2/3 depth. shape (5000, 384), float32. Phase 1 caches in `data/activations.npz` under keys `h0..h5` (raw) and `h0_l2..h5_l2` (L2-normalised).
- Targets: `summary_text` and `summary_lens` from `gpt-4o-mini`. Phase 2 produced 5000 row-aligned summaries (P50 ≈ 36 words) at $0.53 cost.
- AV (Activation Verbalizer): `h3_l2 → text`. Architecture = inject `P_AV(h3_l2)` at one `<ACT>` token of a chat-template prompt; train LoRA-r16 on q+v of frozen 4-bit Qwen2.5-3B-Instruct + the projection. CE loss on the assistant target.
- AR (Activation Reconstructor): `text → h3_l2`. Same shared base, separate LoRA "ar" adapter, plus a fp32 head Linear(2048→384). MSE loss against the L2-normed h3_l2.
- End-of-warm-start metric: **FVE = 1 − mean(‖h − AR(AV(h))‖²) / Var(h)** on a 500-row held-out partition. Doc target ≥ 0.25; hard-fail < 0.10.

The "go to Phase 4" gate was: best variant FVE ≥ 0.10. We never reached it. The blocker was upstream of FVE — AV itself never produced input-correlated text.

---

## 2. Sample data this verdict is based on

### Corpus rows (input to substrate)

```
{"row": 0, "text": "user\nhowdy\nassistant\noh hello. i didn't see you there", "category": "greeting"}
{"row": 1, "text": "user\nwhy are you called guppy\nassistant\nguppy. two syllables. easy", "category": "name"}
{"row": 2, "text": "user\nthe sun is coming through the window\nassistant\ni can see more now. light means daytime…", "category": "light"}
{"row": 3, "text": "user\nare we friends\nassistant\nmy best friend is the filter. it's always there for", "category": "friends"}
{"row": 4, "text": "user\nare you sick\nassistant\nfish get sick when the water is bad. is the", "category": "doctor"}
…
```

5000 rows, 60 categories. Top categories: light (99), name (98), tv (96), night (94), greeting (93). Chance baseline = 99/5000 = 2.0 %.

### Teacher summaries (Phase 2 output)

```
row 0  | summary_text: "The model is likely processing a friendly and casual response to the user's greeting,
                       possibly continuing the conversation with a question or comment to engage the user further."
row 348 | summary_text: "The model is likely processing a continuation about the user's observation of sunlight,
                        expressing a playful or whimsical thought about how daylight leads to the possibility of food …"
row 3448 | summary_text: "The model is likely processing a response about the experience of silence or stillness,
                         possibly reflecting on thoughts or feelings that arise when external noise is absent. …"
```

The teacher's summaries clearly track the input (overlap = 1.75 content words / row on average). Whatever "signal" the activation carries about the input, the teacher captures it.

### AV's generated summaries (after training, on held-out)

Same 8 input texts as above, fed via `h3_l2`/`h3` injection into trained AV:

```
row 348 (light/sun)         AV: "The model is likely processing a response about the concept of seasons, possibly explaining its lack of awareness or understanding of seasonal changes…"
row 3448 (filter stopped)   AV: "The model is likely processing a response about the concept of seasons, possibly explaining its lack of awareness or understanding of seasonal changes…"
row 4248 (are we friends)   AV: "The model is likely processing a response about the concept of seasons, possibly explaining its lack of awareness or understanding of seasonal changes…"
row 3117 (sick fish)        AV: "The model is likely processing a response about the concept of time, possibly discussing its own existence or the nature of its operation as a chatbot…"
row 1999 (bored)            AV: "The model is likely processing a response about the concept of time, possibly discussing its nature or significance…"
row 2031 (love)             AV: "The model is likely processing a response about the concept of seasons, contrasting the user's mention of winter with its own perception of water as a constant element…"
row 364  (love)             AV: "The model is likely processing a response about the concept of time, possibly discussing its own existence or the nature of its operation as a chatbot…"
row 3814 (kids)             AV: "The model is likely processing a response about the concept of seasons, possibly explaining its lack of awareness or understanding of seasonal changes…"
```

Across 8 wildly different inputs (sun, filter, friends, sick fish, bored, love, kids), AV emits **2–3 generic templates** ("seasons / lack of awareness", "time / own existence", "weather"). Content words from the input never appear in AV's output. avg(input ∩ AV) = **0.25 words/row** vs avg(input ∩ teacher) = **1.75** — 7× gap.

This is the smoking gun: AV memorised the *prior* over summary text, not a function of `h_l3`.

### Held-out CE trajectory

```
config        step 250  step 500  step 750  step 1000  step 1250  step 1500  step 1750
L2 + 1×ACT    -         -         1.171     1.170      1.184      1.230      1.380   ↑ overfit
raw + 1×ACT   1.304     -         1.178     -          -          -          -
raw + 8×ACT   1.330     -         1.181     -          -          -          -
zero injection (eval)       1.186 (ckpt: raw best @ step 750)   ← measured by zeroing h before forward
```

All three converge to the same eval CE ≈ 1.18. Variance across checkpoints ≤ 0.01.

---

## 3. Iteration log

### iter 0 — Phase 1 + 2 outputs (pre-existing)

- `data/corpus.jsonl` 5000 rows aligned with substrate ids
- `data/activations.npz` keys `h0..h5`, `h0_l2..h5_l2`, each shape (5000, 384)
- `data/summaries.jsonl` 5000 rows × {summary_text, summary_lens, logit_lens_top3}, $0.53 cost, 0 errors

### iter 1 — Smoke (NaN bug, FIXED)

`scripts/runpod_smoke.sh`: 50 steps AV + 50 steps AR + 16-row joint FVE. RTX A5000, 4-bit Qwen.

- **First run**: `FVE = nan, MSE = nan, h_hat_norm = nan, summary = '!!!!!'` (token id 0 = `!`).
- **RCA (`scripts/debug_step_trace.py`)**: AV.proj and AR.head inherited the embedding's dtype (fp16) via the auto-skip-uint8 heuristic. AdamW state in fp16 underflows: `exp_avg_sq ≈ 1e-9 < fp16 min subnormal (6e-8) → 0`; `eps=1e-8 → 0`; `denom = √v + eps = 0`; `update = lr·m / 0 = ±Inf`. After **one** AdamW step `ar.head.weight` had 714 NaN + 782 455 Inf out of 786 432 elements.
- **Fix (commit `3b1a6c5`)**: hardcode `proj` and `head` to fp32 (~3 MB total trainable). Autocast handles bf16 cast at forward; AdamW state stays fp32.
- **Second run**: dtype mismatch in `ar.head` during FVE eval — `last_token` from frozen base is fp16, head is now fp32 → `F.linear` raised `mat1 and mat2 must have the same dtype`. Fixed by casting `last_token.to(self.head.weight.dtype)`.
- **Third run**: OOM at batch=16 — bf16 autocast still upcasts logits to fp32 inside HF's `loss_function` (1.25 GB allocation). Loosened auto-batch heuristic: `reserved_gb 3 → 5`, `train_per_unit_mb 700 → 1100`. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` added.
- **Fourth run**: ✅ smoke OK. AV CE 2.44, AR MSE 2.08, FVE = −1.19 (expected at 50 steps with 200-step lr-warmup), `h_hat_norm = 1.16`, summary = "The model is likely processing a request or a question about a specific topic, possibly related to a place or event…". Greenlit for full training.

### iter 2 — Full training, L2 + 1× ACT (UNDERTRAINED then OVERFIT)

`scripts/runpod_full.sh` with `--max-steps 10000 --time-budget-min 360 --batch 12 --eval-every 250 --ckpt-every 250`.

Live tqdm postfix added: `loss=<EMA> gn=<grad-norm> lr=<lora-lr> eval=<last eval_ce> gpu=<reserved GB>`.

| Step | eval_ce | train_ema | gn | gpu | observation |
|------|---------|-----------|----|-----|-------------|
| 750  | 1.171   | ~0.83     | ~1.4 | 12 G  | first eval |
| 1000 | 1.170   | ~0.83     | ~1.3 | 17 G  | best — flat from 750 |
| 1250 | 1.184   | 0.78      | 1.4 | 20 G  | rising — overfit begins |
| 1500 | 1.230   | 0.73      | 1.5 | 20 G  | clear overfit |
| 1750 | 1.380   | 0.55      | 1.9 | 20 G  | accelerating |
| 2000 | 1.436   | 0.48      | 1.9 | 20 G  | killed |

Train-EMA still dropping; held-out CE rising → memorising training summaries from the prompt prior, not learning from `h_l3`.

Resume support added (`load_latest_checkpoint`, `start_step`) so we recover from the latest `step_<n>.pt` after kill (commit `26dbcfc`). Best.pt rotation-protected (commit `f58b3e3`). Kept-last-3 step rotation (commit `301fe1f`).

Cost: ~$1.20 on A5000 @ $0.28/hr.

### iter 3 — Diagnostic A: do AV outputs track input?

`scripts/diagnose_av.py` — decode AV(h_l3) for 8 held-out rows, compare to teacher summary + input text bag-of-words overlap.

```
avg(input ∩ AV)     = 0.25 content words / row
avg(input ∩ teacher) = 1.75 content words / row
```

AV produced 2–3 generic templates ("seasons", "time", "weather") regardless of input. **AV is not reading `h_l3`.**

### iter 4 — Substrate probe (DATA HAS SIGNAL)

`scripts/probe_substrate.py` — sklearn LogisticRegression from each layer key → category (60 classes, chance = 2.0%).

```
layer    eval_acc   chance   lift
h2       0.980      0.020    +0.98
h3       0.974      0.020    +0.97   ← raw, ABOVE CHANCE
h1       0.968      0.020    +0.97
h4       0.960      0.020    +0.96
h5       0.958      0.020    +0.96
h0       0.760      0.020    +0.76
h2_l2    0.864      0.020    +0.86
h3_l2    0.826      0.020    +0.82   ← what AV was fed
h1_l2    0.844      0.020    +0.84
h4_l2    0.776      0.020    +0.77
h5_l2    0.730      0.020    +0.73
h0_l2    0.606      0.020    +0.60
```

The substrate **does** carry strong category-level signal. L2-normalisation costs ~15 pts. The data is not the problem.

### iter 5 — Raw h3 (no improvement)

Hypothesis: L2-norm destroys magnitude info AV needs. Added `--raw-h` flag; reads `h3` instead of `h3_l2`.

Result: eval CE @ step 750 = **1.178** (vs L2's 1.171). Within noise.

### iter 6 — Raw h3 + 8× ACT injection (no improvement)

Hypothesis: a single `<ACT>` position is too easy for the deep model to ignore. Repeated `<ACT>` 8× in the user message; `_inject` already broadcasts the projected activation across all `act_id`-matching positions via `torch.where`.

Result: eval CE @ step 750 = **1.181** (essentially the same). No measurable benefit.

### iter 7 — Floor diagnostic (DEFINITIVE)

`scripts/diagnose_floor.py` — load best raw AV; eval CE on 240 held-out rows in two modes:

```
CE WITH    activation : 1.1938
CE WITHOUT activation : 1.1857  (h zeroed before injection)
delta                 : -0.0080
```

**Zeroing the activation gives a slightly *lower* CE than feeding the real one.** The model's loss on held-out is governed entirely by the chat-template prompt prior. The activation contributes **nothing** to free-text generation. This is the conclusive evidence.

---

## 4. Conclusion

The architecture-as-specified does not work on this substrate. Specifically:

- A 384-d activation injected at 1–8 embedding positions of a 130-token chat-template prompt cannot override a 3B-parameter pretrained Qwen's prior over summary-style English, given LoRA-r16 on q+v projections only.
- The CE landscape AV faces is dominated by the prompt prior. AV's gradient w.r.t. P_AV is overwhelmed by the gradient w.r.t. the LoRA's prior-fitting, so P_AV never learns a useful map.
- A linear classifier extracts category from the same activation at 97% accuracy. The *information* is there; the AV training objective just doesn't surface it.

We did NOT test (any of these could change the verdict):

- **Larger LoRA**: r=64 + targets {q,k,v,o,gate_proj,up_proj,down_proj}. Doubles trainable params; might learn an "attend-to-ACT" circuit. ~$1–2 to test.
- **Unfrozen LM head**: lets the model directly map activation-attended hidden states to summary tokens. Currently tied to the embedding (which is frozen via 4-bit). ~$1.
- **Activation as preamble of N projected tokens** (not just N copies of the same vector): `Linear(384, N·hidden)` reshape into N tokens of distinct content. More expressive than the N-copy variant we tried.
- **Smaller decoder**: a 0.5B Qwen with the same LoRA might be easier to override; or matching the substrate scale (9M decoder) might just work.
- **Direct AR pre-training only**: skip AV, train AR alone on `(teacher_summary → h3_l2)`. If AR learns, the inverse problem is feasible and Phase 4 RL could optimise AV jointly via reward.
- **Joint Phase 4 RL from scratch**: skip warm-start entirely.

## 5. Recommendations for any successor work

1. **First check whether the architecture is the bottleneck or the scale**. A control experiment: train AV to predict the substrate's INPUT TEXT (not summary) given `h`. Input text is shorter, simpler, directly observable. If AV cannot learn even that, the architecture is the problem. If it can, the summary-distribution prior is the problem and we need a smaller decoder or richer injection.

2. **Don't fight the prior. Reduce its dominance.** Either use a much smaller decoder (so its prior is weaker), or replace the chat template with a minimal prefix that gives the model less to fall back on. A bare "Activation: <ACT>×N. Summary:" might leave the model nowhere to hide.

3. **Probe results give a free upper bound**. Linear probe lift ≈ 0.97 on 60-way category. Anything lower than ~0.7 should be considered "AV is not extracting any signal".

4. **The floor diagnostic (`scripts/diagnose_floor.py`) is the definitive test.** Run it after every meaningful training run. delta ≥ 0.10 is a working AV; delta < 0.05 means you're at the prior.

## 6. Artifacts

| Path | Purpose |
|------|---------|
| `nla/qwen.py` | base loader; bf16 amp dtype, FA2 detect, fp32 trainable proj/head, 50 GB-aware auto-batch |
| `nla/av.py` | AV(nn.Module) — projection + base + injection-at-`<ACT>`. proj is fp32 |
| `nla/ar.py` | AR(nn.Module) — base + last-token head. head is fp32; explicit dtype-cast at forward |
| `nla/data_phase3.py` | AV/AR datasets + collates. Left-pad. 8× `<ACT>` repeats in user content |
| `nla/splits.py` | deterministic 4500/500 partition (seed=42) |
| `nla/train_warmstart.py` | train_av / train_ar with EarlyStopper, resume support, best.pt + step-rotation, live tqdm postfix |
| `nla/fve.py` | joint AV→AR FVE pipeline |
| `scripts/03_warmstart.py` | CLI driver. `--raw-h` flag |
| `scripts/runpod_*.sh` | smoke + full launchers |
| `scripts/probe_substrate.py` | sklearn LR probe from `h_l → category` per layer |
| `scripts/diagnose_av.py` | decode AV for held-out rows; word-overlap with teacher + input |
| `scripts/diagnose_floor.py` | eval CE with vs without activation injection |
| `scripts/debug_nan_trace.py`, `debug_step_trace.py` | tracers used to localise the AdamW underflow |
| `tests/test_av_ar.py` | 13 unit tests; all pass on the final code |

## 7. Cost summary

| Run | Wall-clock | A5000 cost @ $0.28/hr |
|-----|-----------|----------------------|
| Smoke (4 attempts incl. fixes) | ~30 min | $0.14 |
| Full L2 (steps 0–2000) | ~80 min | $0.37 |
| Diagnostics (av, probe, floor) | ~10 min | $0.05 |
| Raw h3 (steps 0–800) | ~25 min | $0.12 |
| Raw h3 + 8× ACT (steps 0–775) | ~25 min | $0.12 |
| Tooling/idle | ~30 min | $0.14 |
| **Total** | **~3 hr** | **~$0.94** |

Out of $9.12 compute budget, $0.94 spent. ~$8 remaining if a successor experiment is desired (any of the four "we did NOT test" hypotheses fits comfortably).

## 8. Git lineage

Commits this phase (all on `main`, pushed to `origin`):

```
83db9de  feat(phase3): inject activation at 8 positions, not 1
a014a9f  feat(phase3): --raw-h flag to skip L2-normalization
f58b3e3  feat(phase3): always preserve best.pt across rotation
4050278  tune(phase3): batch=12 (override auto-batch); show GPU memory_reserved
2138bd0  feat(phase3): live tqdm postfix with loss EMA + grad-norm + LR + eval + GPU
301fe1f  feat(phase3): keep last 3 step_<n>.pt to cap checkpoint disk
26dbcfc  feat(phase3): resume from latest step_<n>.pt; cap full-run budget to 6h/variant
3b1a6c5  fix(phase3): NaN smoke — fp32 trainable proj/head + dtype cast + tighter auto-batch
1e676b8  fix: bf16 autocast on Ampere+ to prevent fp16 overflow → NaN
04cf014  feat: scripts/runpod_sync.sh — rsync pod outputs to ./pod-sync/
a2ef2bf  fix: AV/AR must skip uint8 params when picking proj/head dtype
24b7a36  perf: use uv on pod for 5-10× faster dep install
6d4fdee  fix: pin transformers<4.50 + peft<0.14 on pod (avoid torch.int1 transitive)
```

Tags applied: none. Phase 3 was never tagged `phase3-complete` because it never met the FVE ≥ 0.10 hard-fail threshold.
