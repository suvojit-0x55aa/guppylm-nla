#!/usr/bin/env bash
# Pull pod outputs to ./pod-sync/ on the local machine.
# Idempotent: re-run any time, only changed files transfer.
set -euo pipefail

POD_HOST="${POD_HOST:-69.30.85.203}"
POD_PORT="${POD_PORT:-22101}"
KEY="${KEY:-$HOME/.ssh/id_ed25519}"
LOCAL_DIR="${LOCAL_DIR:-./pod-sync}"

mkdir -p "$LOCAL_DIR"

SSH_OPTS=(-p "$POD_PORT" -i "$KEY"
          -o StrictHostKeyChecking=no
          -o UserKnownHostsFile=/dev/null)

# Outputs (small, always sync)
echo "==> sync /workspace/out → $LOCAL_DIR/out"
rsync -avz --delete \
    -e "ssh ${SSH_OPTS[*]}" \
    root@"$POD_HOST":/workspace/out/ \
    "$LOCAL_DIR/out/" \
    2>&1 | tail -10

# Optional: checkpoints (~50-200 MB; skip by default)
if [[ "${SYNC_CKPTS:-0}" == "1" ]]; then
    echo "==> sync /workspace/ckpts → $LOCAL_DIR/ckpts"
    rsync -avz \
        -e "ssh ${SSH_OPTS[*]}" \
        root@"$POD_HOST":/workspace/ckpts/ \
        "$LOCAL_DIR/ckpts/" \
        2>&1 | tail -5
fi

echo
echo "==> contents:"
find "$LOCAL_DIR" -type f -not -path '*/\.*' | head -20
