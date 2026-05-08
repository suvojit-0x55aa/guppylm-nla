#!/usr/bin/env bash
# Download arman-bd/guppylm-9M from HuggingFace, pin SHA, place files where
# the inference loader expects them.
set -euo pipefail

cd "$(dirname "$0")/.."

REPO="arman-bd/guppylm-9M"
PY="${PYTHON:-.venv/bin/python}"
[[ -x "$PY" ]] || { echo "venv python not found at $PY"; exit 1; }

mkdir -p checkpoints data

"$PY" - <<'PYEOF'
import json, os, shutil
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi

repo = "arman-bd/guppylm-9M"
api = HfApi()
info = api.repo_info(repo_id=repo)
sha = info.sha
print(f"HF revision: {sha}")

local = snapshot_download(
    repo_id=repo,
    revision=sha,
    local_dir="checkpoints/guppylm-9M",
    allow_patterns=["pytorch_model.bin", "config.json", "tokenizer.json"],
)
print(f"Downloaded to: {local}")

ckpt_dir = Path("checkpoints")
data_dir = Path("data")
src = Path(local)

# inference loader expects:
#   checkpoints/best_model.pt
#   checkpoints/config.json
#   data/tokenizer.json
shutil.copy2(src / "pytorch_model.bin", ckpt_dir / "best_model.pt")
shutil.copy2(src / "config.json", ckpt_dir / "config.json")
shutil.copy2(src / "tokenizer.json", data_dir / "tokenizer.json")
(ckpt_dir / "REVISION").write_text(sha + "\n")

for p in [ckpt_dir / "best_model.pt", ckpt_dir / "config.json", data_dir / "tokenizer.json"]:
    print(f"  {p}  ({p.stat().st_size/1e6:.1f} MB)")

# --- Source corpus: download the synthetic fish-chat dataset from HF ---
# arman-bd/guppylm-60k-generic ships parquet only; convert to JSONL for the
# Phase-1 extractor (no pyarrow runtime dep).
import struct
ds_local = snapshot_download(
    repo_id="arman-bd/guppylm-60k-generic",
    repo_type="dataset",
    local_dir="checkpoints/guppylm-60k-generic",
    allow_patterns=["data/train-00000-of-00001.parquet"],
)
parquet = Path(ds_local) / "data" / "train-00000-of-00001.parquet"
print(f"Dataset: {parquet}")

# Convert parquet -> jsonl. pandas+pyarrow is heavy; install pyarrow on demand.
import pyarrow.parquet as pq

table = pq.read_table(str(parquet))
print(f"Parquet columns: {table.column_names}, rows: {table.num_rows}")
out = data_dir / "train.jsonl"
# The HF dataset ships raw {input, output, category} rows. The model was
# pretrained on the formatted chat template (generate_data.format_sample) —
# we apply the same formatting so activations match the substrate's
# input distribution.
def _format(s):
    return (
        f"<|im_start|>user\n{s['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{s['output']}<|im_end|>"
    )
with open(out, "w") as f:
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    n = table.num_rows
    for i in range(n):
        row = {k: cols[k][i] for k in cols}
        f.write(json.dumps({"text": _format(row), "category": row.get("category")}) + "\n")
print(f"  {out}  ({out.stat().st_size/1e6:.1f} MB, {n} rows)")
print("Done.")
PYEOF
