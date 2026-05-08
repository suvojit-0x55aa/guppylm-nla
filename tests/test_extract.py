"""Phase 1 sanity gates. Runs against the smoke fixture (n=100).

Auto-generates the smoke fixture on first run if missing — keeps the
test suite self-contained but reuses cached output if present.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SMOKE_CORPUS = ROOT / "data" / "corpus_smoke.jsonl"
SMOKE_NPZ = ROOT / "data" / "activations_smoke.npz"
SMOKE_MANIFEST = ROOT / "data" / "MANIFEST_smoke.json"


def _generate_smoke_if_missing():
    if SMOKE_CORPUS.exists() and SMOKE_NPZ.exists():
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "01_extract_activations.py"),
        "--n", "100",
        "--out-corpus", str(SMOKE_CORPUS),
        "--out-activations", str(SMOKE_NPZ),
        "--out-manifest", str(SMOKE_MANIFEST),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


@pytest.fixture(scope="module")
def smoke():
    _generate_smoke_if_missing()
    npz = np.load(SMOKE_NPZ)
    rows = [json.loads(l) for l in open(SMOKE_CORPUS)]
    return {"npz": npz, "rows": rows}


def test_shapes(smoke):
    npz = smoke["npz"]
    n = len(smoke["rows"])
    for l in range(6):
        for key in (f"h{l}", f"h{l}_l2"):
            arr = npz[key]
            assert arr.shape == (n, 384), f"{key} shape {arr.shape} != ({n}, 384)"
            assert arr.dtype == np.float32, f"{key} dtype {arr.dtype} != float32"


def test_finite(smoke):
    npz = smoke["npz"]
    for key in npz.files:
        assert np.isfinite(npz[key]).all(), f"non-finite values in {key}"


def test_l2_unit_norm(smoke):
    npz = smoke["npz"]
    for l in range(6):
        norms = np.linalg.norm(npz[f"h{l}_l2"], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"h{l}_l2 not unit-norm: range [{norms.min()}, {norms.max()}]"


def test_layer_distinct(smoke):
    """Adjacent-layer residuals should not be near-identical (would mean hooks aliased)."""
    npz = smoke["npz"]
    for i in range(5):
        a = npz[f"h{i}_l2"]
        b = npz[f"h{i+1}_l2"]
        cos = (a * b).sum(axis=1).mean()
        assert cos < 0.99, f"cos(h{i}, h{i+1}) = {cos:.4f} ≥ 0.99 — hook aliasing?"


def test_corpus_alignment(smoke):
    npz = smoke["npz"]
    n = len(smoke["rows"])
    for key in npz.files:
        assert npz[key].shape[0] == n, f"{key} rows ({npz[key].shape[0]}) != corpus rows ({n})"
    # Row indices should be 0..n-1
    for i, r in enumerate(smoke["rows"]):
        assert r["row"] == i, f"row[{i}] has row-index {r['row']}"


def test_logit_lens_plausibility(smoke):
    """Top-1 logit-lens prediction at h5 should be a non-pad, valid vocab token."""
    from nla.load import load_substrate
    model, tokenizer, _ = load_substrate(
        str(ROOT / "checkpoints" / "best_model.pt"),
        str(ROOT / "data" / "tokenizer.json"),
    )
    npz = smoke["npz"]
    n = len(smoke["rows"])
    pad_id = model.config.pad_id
    sample_indices = list(range(0, min(5, n)))
    with torch.no_grad():
        for idx in sample_indices:
            h = torch.tensor(npz["h5"][idx], dtype=torch.float32).unsqueeze(0)
            logits = model.lm_head(model.norm(h))[0]
            top1 = int(torch.argmax(logits).item())
            assert top1 != pad_id, f"row {idx} top-1 is <pad>"
            decoded = tokenizer.decode([top1])
            assert isinstance(decoded, str) and len(decoded) >= 0
