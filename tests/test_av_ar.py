"""Phase 3 unit tests — tiny random Qwen2, no API/network spend."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _tiny_qwen():
    """Untrained Qwen2 with tiny dims; runs instantly on CPU."""
    cfg = Qwen2Config(
        vocab_size=151936,                    # full Qwen vocab so the real tokenizer works
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg).eval()
    return model


@pytest.fixture(scope="module")
def base_and_tok():
    base = _tiny_qwen()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": ["<ACT>"]})
    base.resize_token_embeddings(len(tok), mean_resizing=False)
    act_id = tok.convert_tokens_to_ids("<ACT>")
    return base, tok, act_id


# ── AV ────────────────────────────────────────────────────────────────────────


def test_av_inject_localized(base_and_tok):
    from nla.av import AV
    base, tok, act_id = base_and_tok
    av = AV(base, act_id, d_substrate=384, lora_r=4, lora_alpha=8)
    B, T = 2, 6
    input_ids = torch.tensor([
        [1, 2, act_id, 4, 5, 6],
        [1, 2, 3, act_id, 5, 6],
    ], dtype=torch.long)
    h = torch.randn(B, 384)
    emb_before = av.base.get_input_embeddings()(input_ids)
    emb_after = av._inject(input_ids, h)
    # Non-<ACT> rows unchanged.
    for b in range(B):
        for t in range(T):
            if input_ids[b, t] != act_id:
                assert torch.allclose(emb_after[b, t], emb_before[b, t]), \
                    f"row {b} pos {t} should be unchanged"
    # <ACT> rows replaced.
    inj = av.proj(h)
    for b in range(B):
        t_act = (input_ids[b] == act_id).nonzero()[0, 0]
        assert torch.allclose(emb_after[b, t_act].float(), inj[b].float(), atol=1e-3)


def test_av_inject_bf16_base_fp32_input(base_and_tok):
    """Regression: on T4/Qwen the base loads in bf16; h_l from the dataloader is fp32.
    AV's proj is always fp32 (so AdamW state stays fp32 — fp16 underflows during
    the optimizer step), and _inject casts the proj output back to the base's
    embedding dtype before injection."""
    from nla.av import AV
    base, tok, act_id = base_and_tok
    base_bf = base.to(torch.bfloat16)
    av = AV(base_bf, act_id, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="av_bf")
    assert av.proj.weight.dtype == torch.float32
    B, T = 1, 4
    input_ids = torch.tensor([[act_id, 1, 2, 3]], dtype=torch.long)
    h = torch.randn(B, 384, dtype=torch.float32)        # mimic dataloader fp32 input
    emb = av._inject(input_ids, h)
    assert torch.isfinite(emb).all()
    assert emb.dtype == torch.bfloat16                   # matches base embedding dtype
    base.to(torch.float32)                               # reset for downstream tests


def test_av_forward_shape(base_and_tok):
    from nla.av import AV
    base, tok, act_id = base_and_tok
    av = AV(base, act_id, d_substrate=384, lora_r=4, lora_alpha=8)
    B, T = 2, 8
    input_ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    input_ids[:, 0] = act_id
    attn = torch.ones((B, T), dtype=torch.long)
    h = torch.randn(B, 384)
    out = av(input_ids=input_ids, attention_mask=attn, h_l=h, labels=None)
    assert out.logits.shape == (B, T, base.config.vocab_size)


def test_av_generate_stops_on_eos(base_and_tok):
    from nla.av import AV
    base, tok, act_id = base_and_tok
    av = AV(base, act_id, d_substrate=384, lora_r=4, lora_alpha=8)
    B, T = 1, 4
    input_ids = torch.tensor([[act_id, 100, 101, 102]], dtype=torch.long)
    attn = torch.ones((B, T), dtype=torch.long)
    h = torch.randn(B, 384)
    out_ids = av.generate(
        input_ids=input_ids, attention_mask=attn, h_l=h,
        max_new_tokens=5, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
    )
    # Returns full sequence: prompt + new tokens.
    assert out_ids.shape[0] == B
    assert out_ids.shape[1] >= T + 1
    assert out_ids.shape[1] <= T + 5


# ── AR ────────────────────────────────────────────────────────────────────────


def test_ar_forward_shape(base_and_tok):
    from nla.ar import AR
    base, tok, act_id = base_and_tok
    ar = AR(base, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="ar_only")
    B, T = 2, 10
    input_ids = torch.randint(0, base.config.vocab_size, (B, T), dtype=torch.long)
    attn = torch.ones((B, T), dtype=torch.long)
    out = ar(input_ids=input_ids, attention_mask=attn)
    assert out.shape == (B, 384)
    assert out.dtype == torch.float32     # AR returns fp32 for stable MSE


def test_ar_zero_init(base_and_tok):
    """Regression: AR.head must be zero-initialized so initial ĥ ≈ 0 and MSE
    against unit-norm targets stays at ~1 (not ~440). Otherwise smoke FVE
    starts at -300+ and training has to dig out of a deep hole."""
    from nla.ar import AR
    base, tok, _ = base_and_tok
    ar = AR(base, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="ar_init")
    assert ar.head.weight.abs().max().item() == 0.0
    assert ar.head.bias.abs().max().item() == 0.0
    B, T = 4, 12
    input_ids = torch.randint(0, base.config.vocab_size, (B, T), dtype=torch.long)
    attn = torch.ones((B, T), dtype=torch.long)
    out = ar(input_ids=input_ids, attention_mask=attn)
    assert out.abs().max().item() < 1e-5, f"zero-init violated: max abs = {out.abs().max().item()}"


def test_ar_uses_last_non_pad_token(base_and_tok):
    """AR should use the last *non-padded* token's hidden state, not the literal last position."""
    from nla.ar import AR
    base, tok, act_id = base_and_tok
    ar = AR(base, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="ar_pad_test")
    B, T = 2, 10
    input_ids = torch.full((B, T), tok.pad_token_id, dtype=torch.long)
    input_ids[0, :4] = torch.tensor([10, 20, 30, 40])
    input_ids[1, :7] = torch.tensor([10, 20, 30, 40, 50, 60, 70])
    attn = (input_ids != tok.pad_token_id).long()
    out = ar(input_ids=input_ids, attention_mask=attn)
    assert out.shape == (B, 384)
    assert torch.isfinite(out).all()


# ── FVE ───────────────────────────────────────────────────────────────────────


def test_fve_synthetic():
    """Hand-checked FVE: var(h) and constant predictor."""
    from nla.fve import variance_of_targets
    rng = np.random.default_rng(0)
    h = rng.normal(size=(50, 8)).astype(np.float32)
    h_var = variance_of_targets(h)
    # Constant predictor (mean) achieves MSE = sum(var(h)) → FVE = 0.
    h_hat = np.broadcast_to(h.mean(axis=0), h.shape)
    mse = float(((h - h_hat) ** 2).sum(axis=-1).mean())
    fve = 1.0 - mse / h_var
    assert abs(fve) < 1e-3


# ── Splits ────────────────────────────────────────────────────────────────────


def test_split_determinism():
    from nla.splits import make_or_load_split
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "splits.json"
        a_train, a_eval = make_or_load_split(1000, eval_size=100, seed=42, path=p)
        b_train, b_eval = make_or_load_split(1000, eval_size=100, seed=42, path=p)
        assert a_train == b_train
        assert a_eval == b_eval
        assert len(a_train) + len(a_eval) == 1000
        assert set(a_train).isdisjoint(set(a_eval))


def test_dataset_variant_short_and_long_keys(base_and_tok):
    """Regression: the CLI passes 'text'/'lens' but JSONL has 'summary_text'/'summary_lens'.
    Both forms must work."""
    from nla.data_phase3 import AVDataset, ARDataset
    _, tok, _ = base_and_tok
    summaries = [
        {"row": 0, "summary_text": "alpha summary.", "summary_lens": "alpha lens."},
        {"row": 1, "summary_text": "beta summary.",  "summary_lens": "beta lens."},
    ]
    h = np.zeros((2, 384), dtype=np.float32)
    for short, full, expected_in_target in [("text", "summary_text", "alpha summary"),
                                            ("lens", "summary_lens", "alpha lens")]:
        for variant in (short, full):
            ds_av = AVDataset([0, 1], summaries, h, tok, variant=variant)
            item = ds_av[0]
            decoded = tok.decode(item["input_ids"].tolist())
            assert expected_in_target in decoded, f"variant={variant}: missing target in {decoded[:200]!r}"
            ds_ar = ARDataset([0, 1], summaries, h, tok, variant=variant)
            ar_item = ds_ar[0]
            ar_decoded = tok.decode(ar_item["input_ids"].tolist())
            assert expected_in_target in ar_decoded


def test_proj_head_skip_uint8_params_from_bnb_base(base_and_tok):
    """Regression: bnb 4-bit packs weights as torch.uint8. The first param
    enumerated on a bnb-wrapped base is uint8, and `nn.Linear(dtype=uint8)`
    produces garbage. AV.proj and AR.head must skip integer-typed params and
    fall back to a real floating-point dtype."""
    from nla.av import AV
    from nla.ar import AR
    base, tok, act_id = base_and_tok
    # Inject a leading uint8 param (mimics bnb's Params4bit flat byte buffer).
    base.register_buffer("_fake_packed_q_weight", torch.zeros(8, dtype=torch.uint8),
                         persistent=False)
    # nn.Module.parameters() doesn't yield buffers, so register as a param too.
    if not hasattr(base, "_fake_packed_q_weight_p"):
        base.register_parameter(
            "_fake_packed_q_weight_p",
            torch.nn.Parameter(torch.zeros(8, dtype=torch.uint8), requires_grad=False),
        )
    av = AV(base, act_id, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="av_uint8")
    assert av.proj.weight.dtype.is_floating_point, f"proj got {av.proj.weight.dtype}"
    ar = AR(av.base, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="ar_uint8")
    assert ar.head.weight.dtype.is_floating_point, f"head got {ar.head.weight.dtype}"


def test_checkpoint_roundtrip_preserves_proj_and_head(base_and_tok):
    """Regression: save_final/load_final must persist P_AV (proj) and Q_AR (head)
    weights, not just LoRA. Earlier filter used '.proj.' and '.head.' literal
    substrings — those don't match top-level keys 'proj.weight' / 'head.weight',
    so the projection/head silently dropped during save."""
    from nla.av import AV
    from nla.ar import AR
    from nla.train_warmstart import save_final, load_final, _trainable_state
    base, tok, act_id = base_and_tok
    av = AV(base, act_id, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="av_ckpt")
    # Mutate proj weights so default-init doesn't accidentally hide a bug
    with torch.no_grad():
        av.proj.weight.copy_(torch.randn_like(av.proj.weight) * 0.5)
        av.proj.bias.copy_(torch.full_like(av.proj.bias, 0.123))
    state = _trainable_state(av)
    proj_keys = [k for k in state if 'proj' in k]
    assert proj_keys, f"proj.* missing from saved state. keys: {list(state)[:5]}"

    ar = AR(av.base, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="ar_ckpt")
    with torch.no_grad():
        ar.head.weight.copy_(torch.randn_like(ar.head.weight) * 0.5)
        ar.head.bias.copy_(torch.full_like(ar.head.bias, 0.456))
    state_ar = _trainable_state(ar)
    head_keys = [k for k in state_ar if 'head' in k]
    assert head_keys, f"head.* missing from saved state. keys: {list(state_ar)[:5]}"

    # Round-trip via save_final/load_final
    with tempfile.TemporaryDirectory() as d:
        save_final(av, Path(d), step=10)
        # Build a fresh AV with default init (proj should differ)
        av2 = AV(base, act_id, d_substrate=384, lora_r=4, lora_alpha=8, adapter_name="av_ckpt2")
        info = load_final(av2, Path(d), device="cpu")
        assert info is not None
        assert torch.allclose(av2.proj.weight.float(), av.proj.weight.float(), atol=1e-4)
        assert torch.allclose(av2.proj.bias.float(), av.proj.bias.float(), atol=1e-4)


def test_split_seed_changes_partition():
    from nla.splits import make_or_load_split
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "splits1.json"; p2 = Path(d) / "splits2.json"
        _, eval_42 = make_or_load_split(1000, eval_size=100, seed=42, path=p1)
        _, eval_7 = make_or_load_split(1000, eval_size=100, seed=7, path=p2)
        assert eval_42 != eval_7
