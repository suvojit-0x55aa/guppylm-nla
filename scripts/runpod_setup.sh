#!/usr/bin/env bash
# One-time pod setup: install deps + untar bundle + sanity-check GPU.
# Idempotent: safe to re-run after `git pull`.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/guppylm-nla}"
WORKSPACE="${WORKSPACE:-/workspace}"
BUNDLE="${BUNDLE:-$WORKSPACE/guppylm-nla-bundle.tar.gz}"
HF_HOME_DIR="${HF_HOME:-$WORKSPACE/hf_cache}"

echo "==> repo:       $REPO_DIR"
echo "==> workspace:  $WORKSPACE"
echo "==> bundle:     $BUNDLE"
echo "==> hf cache:   $HF_HOME_DIR"

# 1. Sanity-check GPU.
echo
echo "==> nvidia-smi"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# 2. Install uv (5-10× faster than pip), then install deps with it.
if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo
echo "==> uv pip install (system python)"
# Pin transformers and peft to pre-torchao-strict versions. The pod's
# preinstalled torch is older than 2.5, so anything pulling in torchao>=0.16
# (which uses torch.int1) explodes at import. transformers <4.50 + peft <0.14
# don't import torchao for our LoRA path, and bitsandbytes 4-bit still works.
uv pip install --system -q \
    'transformers>=4.43,<4.50' 'peft>=0.12,<0.14' \
    'accelerate>=0.30' 'datasets>=2.20' 'bitsandbytes>=0.43' \
    'numpy>=1.24' 'tqdm>=4.65' 'tokenizers>=0.19' \
    'huggingface_hub>=0.20' 'pyarrow>=14' \
    'pytest>=7.4' 'pytest-asyncio>=0.21' 'openai>=1.40'

# 3. Untar bundle (data + checkpoints) into the cloned repo, only if missing.
if [[ ! -f "$BUNDLE" ]]; then
    echo
    echo "ERROR: bundle not found at $BUNDLE"
    echo "scp it from your laptop:"
    echo "  scp guppylm-nla-bundle.tar.gz root@<pod>:$WORKSPACE/"
    exit 1
fi

cd "$REPO_DIR"
if [[ ! -f data/corpus.jsonl ]]; then
    echo
    echo "==> extracting bundle"
    STAGE="$WORKSPACE/_bundle_stage"
    rm -rf "$STAGE" && mkdir -p "$STAGE"
    # --no-same-owner: bundle was tarred on macOS with uid 501; pod runs as
    #   root and would error trying to chown to a non-existent uid.
    # --exclude='._*': drop AppleDouble resource forks the macOS tar embedded.
    tar xzf "$BUNDLE" -C "$STAGE" --no-same-owner --exclude='._*'
    for sub in data checkpoints; do
        if [[ -d "$STAGE/$sub" ]]; then
            rm -rf "$REPO_DIR/$sub"
            cp -r "$STAGE/$sub" "$REPO_DIR/$sub"
            echo "    synced $sub/"
        fi
    done
    rm -rf "$STAGE"
fi

# 4. HF cache on persistent volume.
mkdir -p "$HF_HOME_DIR"

# 5. Tests pass before any real training.
echo
echo "==> pytest"
HF_HOME="$HF_HOME_DIR" python -m pytest tests/test_av_ar.py -q

echo
echo "==> setup OK. next: bash scripts/runpod_smoke.sh"
