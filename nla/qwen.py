"""Qwen base-model loader for Phase 3 AV/AR.

Two paths:
  - 4-bit (bitsandbytes): default on Colab T4, matches doc spec.
  - fp16 fallback: for macOS arm64 where bitsandbytes can't install. Used for
    local plumbing tests with a tiny model id; not for serious training.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ACT_TOKEN = "<ACT>"


def best_amp_dtype() -> torch.dtype:
    """bf16 on Ampere+ (no GradScaler needed); fp16 on older GPUs."""
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def auto_batch_sizes(
    *,
    train_floor: int = 1,
    train_ceiling: int = 16,
    eval_floor: int = 2,
    eval_ceiling: int = 32,
    reserved_gb: float = 5.0,
    train_per_unit_mb: float = 1100.0,
    eval_per_unit_mb: float = 250.0,
) -> tuple[int, int, dict]:
    """Heuristic batch sizes from current free CUDA memory.

    Should be called AFTER load_qwen() so the base model + adapters + KV
    cache scratch are already on the device.

    Constants are tuned empirically for Qwen 3B-4bit + LoRA r=16 at
    seq_len ≈ 128:
      ~700 MB / batch unit during training (fwd+bwd+optimizer state)
      ~200 MB / batch unit during AV.generate eval (KV cache + activations)

    Returns (train_batch, eval_batch, info) — info has the full budget breakdown.
    """
    import torch

    if not torch.cuda.is_available():
        return train_floor, eval_floor, {
            "free_gb": 0.0, "total_gb": 0.0, "device": "cpu/mps", "fallback": True,
        }

    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = free_b / 1024 ** 3
    total_gb = total_b / 1024 ** 3
    available_gb = max(0.0, free_gb - reserved_gb)

    train_b = int(available_gb * 1024 / train_per_unit_mb)
    eval_b = int(available_gb * 1024 / eval_per_unit_mb)

    # Round to powers of 2 (kernels prefer it) but allow non-power for low values.
    def _snap(x: int, lo: int, hi: int) -> int:
        x = max(lo, min(hi, x))
        if x >= 4:
            return 1 << (x.bit_length() - 1)        # largest pow2 ≤ x
        return x

    train_b = _snap(train_b, train_floor, train_ceiling)
    eval_b = _snap(eval_b, eval_floor, eval_ceiling)

    info = {
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "available_gb_after_reserve": round(available_gb, 2),
        "reserved_gb": reserved_gb,
        "device": torch.cuda.get_device_name(0),
    }
    return train_b, eval_b, info


def _best_attn_impl() -> str | None:
    """Prefer flash-attention-2 if importable; falls back to default sdpa.
    sdpa fires a benign sliding-window warning on Qwen2.5 (seq < window so it's
    a no-op) but the SDPA fallback path is slower than FA2 for our 130-token
    seqs. Returns None to let HF pick the default."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return None


def load_qwen(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    use_4bit: bool = True,
    device_map: str | dict | None = "auto",
    dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
):
    """Load (base, tokenizer, act_token_id).

    Args:
        model_id: HF model id. Default Qwen2.5-3B-Instruct.
        use_4bit: bitsandbytes 4-bit nf4 + double-quant. Requires CUDA + bnb.
                  Set False for fp16 fallback (Mac dev or no-bnb Colab).
        device_map: "auto" by default; pass {"": "mps"} or "cpu" for non-CUDA.
        dtype: compute dtype. fp16 on T4 / MPS. bf16 if user prefers on H100.
    """
    if dtype is None:
        dtype = best_amp_dtype()
    quantization_config = None
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    added = tok.add_special_tokens({"additional_special_tokens": [ACT_TOKEN]})

    base_kwargs: dict = {"device_map": device_map}
    if quantization_config is not None:
        base_kwargs["quantization_config"] = quantization_config
    else:
        base_kwargs["dtype"] = dtype
    impl = attn_implementation or _best_attn_impl()
    if impl is not None:
        base_kwargs["attn_implementation"] = impl
        print(f"  attn_implementation: {impl}")

    base = AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs)

    if added > 0:
        base.resize_token_embeddings(len(tok), mean_resizing=False)

    act_token_id = tok.convert_tokens_to_ids(ACT_TOKEN)
    return base, tok, act_token_id
