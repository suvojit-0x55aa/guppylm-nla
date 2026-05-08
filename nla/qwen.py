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


def load_qwen(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    use_4bit: bool = True,
    device_map: str | dict | None = "auto",
    dtype: torch.dtype = torch.float16,
):
    """Load (base, tokenizer, act_token_id).

    Args:
        model_id: HF model id. Default Qwen2.5-3B-Instruct.
        use_4bit: bitsandbytes 4-bit nf4 + double-quant. Requires CUDA + bnb.
                  Set False for fp16 fallback (Mac dev or no-bnb Colab).
        device_map: "auto" by default; pass {"": "mps"} or "cpu" for non-CUDA.
        dtype: compute dtype. fp16 on T4 / MPS. bf16 if user prefers on H100.
    """
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

    base = AutoModelForCausalLM.from_pretrained(model_id, **base_kwargs)

    if added > 0:
        base.resize_token_embeddings(len(tok), mean_resizing=False)

    act_token_id = tok.convert_tokens_to_ids(ACT_TOKEN)
    return base, tok, act_token_id
