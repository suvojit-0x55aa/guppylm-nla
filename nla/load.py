"""Substrate loader. Mirrors arman-bd/guppylm guppylm/inference.py:15-62
so HF-style and legacy checkpoints both work."""

import json
import os
from pathlib import Path

import torch
from tokenizers import Tokenizer

from ._substrate import GuppyConfig, GuppyLM


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _config_from_file(path: Path) -> GuppyConfig:
    with open(path) as f:
        cfg = json.load(f)
    # Support both HF-style and our own keys (verbatim from upstream).
    return GuppyConfig(
        vocab_size=cfg.get("vocab_size", 4096),
        max_seq_len=cfg.get("max_position_embeddings", cfg.get("max_seq_len", 128)),
        d_model=cfg.get("hidden_size", cfg.get("d_model", 384)),
        n_layers=cfg.get("num_hidden_layers", cfg.get("n_layers", 6)),
        n_heads=cfg.get("num_attention_heads", cfg.get("n_heads", 6)),
        ffn_hidden=cfg.get("intermediate_size", cfg.get("ffn_hidden", 768)),
        dropout=cfg.get("hidden_dropout_prob", cfg.get("dropout", 0.1)),
        pad_id=cfg.get("pad_token_id", cfg.get("pad_id", 0)),
        bos_id=cfg.get("bos_token_id", cfg.get("bos_id", 1)),
        eos_id=cfg.get("eos_token_id", cfg.get("eos_id", 2)),
    )


def load_substrate(
    checkpoint_path: str | Path,
    tokenizer_path: str | Path,
    device: str = "cpu",
) -> tuple[GuppyLM, Tokenizer, GuppyConfig]:
    """Load GuppyLM weights + tokenizer. Returns (model.eval(), tokenizer, config)."""
    dev = _resolve_device(device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    ckpt = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
    state_dict = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )

    config_dir = Path(checkpoint_path).resolve().parent
    config_json = config_dir / "config.json"
    if config_json.exists():
        config = _config_from_file(config_json)
    elif isinstance(ckpt, dict) and "config" in ckpt:
        valid = {f for f in GuppyConfig.__dataclass_fields__}
        config = GuppyConfig(**{k: v for k, v in ckpt["config"].items() if k in valid})
    else:
        config = GuppyConfig()

    model = GuppyLM(config).to(dev)
    filtered = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered)
    model.eval()
    return model, tokenizer, config
