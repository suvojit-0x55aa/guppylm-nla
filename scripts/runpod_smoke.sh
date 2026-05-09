#!/usr/bin/env bash
# Smoke run — verifies the full pipeline on the pod's GPU.
# ~3 min on A5000. AV trains 50 steps, AR trains 50 steps, then a 16-row FVE eval.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/guppylm-nla}"
WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME_DIR="${HF_HOME:-$WORKSPACE/hf_cache}"

cd "$REPO_DIR"
mkdir -p "$HF_HOME_DIR"

echo "==> smoke: variant=text, max-steps=50, time-budget=10min, eval=16 rows"
echo "==> ckpts: $WORKSPACE/ckpts/phase3_smoke"
echo "==> out:   $WORKSPACE/out/smoke"

HF_HOME="$HF_HOME_DIR" python scripts/03_warmstart.py \
    --variant text \
    --max-steps 50 --min-steps 0 --time-budget-min 10 \
    --batch 0 --eval-batch 0 \
    --fve-eval-size 16 --final-fve-eval-size 16 \
    --no-skip-if-trained \
    --ckpt-root "$WORKSPACE/ckpts/phase3_smoke" \
    --history-root "$WORKSPACE/out/smoke" \
    --manifest "$WORKSPACE/out/smoke/MANIFEST_phase3_smoke.json"

echo
echo "==> smoke result"
python - <<PYEOF
import json
h = json.load(open("$WORKSPACE/out/smoke/history_text.json"))
print(f"  AV: stop={h['av']['stop_reason']}, steps={h['av']['final_step']}")
print(f"  AR: stop={h['ar']['stop_reason']}, steps={h['ar']['final_step']}")
print(f"  final FVE: {h['final_fve']:.4f}  MSE: {h['final_mse']:.4f}  n={len(h['samples'])}")
if h["samples"]:
    s = h["samples"][0]
    print(f"  sample row 0: h_norm={s['h_norm']:.3f} h_hat_norm={s['h_hat_norm']:.3f} se={s['se']:.3f}")
    print(f"    summary: {s['summary'][:120]!r}")
PYEOF

echo
echo "==> smoke OK if h_hat_norm > 0 and < ~10. Next: bash runpod/full.sh"
