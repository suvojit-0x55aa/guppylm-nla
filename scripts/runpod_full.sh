#!/usr/bin/env bash
# Full Phase 3 — both variants (text + lens), convergence-bounded.
# Time budget set to 24 h to effectively disable the time gate; convergence
# (CE plateau for AV, FVE plateau for AR) decides when to stop.
#
# Re-runs are cheap: --skip-if-trained loads any final.pt that already
# exists in --ckpt-root and skips that phase.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/guppylm-nla}"
WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME_DIR="${HF_HOME:-$WORKSPACE/hf_cache}"
CKPT_ROOT="${CKPT_ROOT:-$WORKSPACE/ckpts/phase3}"
OUT_ROOT="${OUT_ROOT:-$WORKSPACE/out}"
MAX_STEPS="${MAX_STEPS:-10000}"            # plan-aligned step ceiling
TIME_BUDGET_MIN="${TIME_BUDGET_MIN:-360}"  # 6 h / variant (AV+AR) — convergence stopper usually fires earlier

cd "$REPO_DIR"
mkdir -p "$HF_HOME_DIR" "$CKPT_ROOT" "$OUT_ROOT"

run_variant() {
    local variant="$1"
    echo
    echo "================================================================"
    echo "==> $(date -u +%Y-%m-%dT%H:%M:%SZ)  full: variant=$variant"
    echo "==> max-steps=$MAX_STEPS  time-budget=${TIME_BUDGET_MIN}min"
    echo "================================================================"
    HF_HOME="$HF_HOME_DIR" python scripts/03_warmstart.py \
        --variant "$variant" \
        --max-steps "$MAX_STEPS" --min-steps 1000 \
        --time-budget-min "$TIME_BUDGET_MIN" \
        --batch 0 --eval-batch 0 \
        --eval-every 200 --ckpt-every 500 \
        --fve-eval-size 64 --final-fve-eval-size 200 \
        --skip-if-trained \
        --ckpt-root "$CKPT_ROOT" \
        --history-root "$OUT_ROOT" \
        --manifest "$OUT_ROOT/MANIFEST_phase3.json"
}

run_variant text
run_variant lens

echo
echo "==> full run done"
python - <<PYEOF
import json
for v in ("text", "lens"):
    p = "$OUT_ROOT/history_" + v + ".json"
    try:
        h = json.load(open(p))
        print(f"  {v}: AV stop={h['av']['stop_reason']} steps={h['av']['final_step']}  "
              f"AR stop={h['ar']['stop_reason']} steps={h['ar']['final_step']}  "
              f"FVE={h['final_fve']:.4f}")
    except Exception as e:
        print(f"  {v}: error reading {p}: {e}")
PYEOF
echo
echo "==> manifest: $OUT_ROOT/MANIFEST_phase3.json"
echo "==> pull results back with:"
echo "    scp -r root@<pod-host>:$OUT_ROOT ./phase3-out"
