"""Activation Reconstructor (AR).

AR reads a chat-formatted prompt that contains a description of the model's
internal state, runs the AV/AR shared base, takes the final-token hidden
state, and projects it back to the substrate residual space (384-d) via Q_AR.

Frozen: AR base weights (4-bit Qwen). Shared with AV.
Trainable: LoRA adapter "ar" + Q_AR linear.

The shared-base design saves ~3 GB VRAM. PEFT supports multiple adapters on
a single underlying base via `add_adapter` + `set_adapter`. Pass either a
raw base (in which case AR wraps it with PEFT itself) or an already-wrapped
PeftModel from AV (in which case AR just adds its own adapter).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model


class AR(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        d_substrate: int = 384,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        adapter_name: str = "ar",
    ):
        super().__init__()
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(target_modules), task_type="CAUSAL_LM",
        )
        if isinstance(base, PeftModel):
            base.add_adapter(adapter_name, cfg)
            self.base = base
        else:
            self.base = get_peft_model(base, cfg, adapter_name=adapter_name)
        self.adapter_name = adapter_name

        d_hidden = self.base.config.hidden_size
        head_dtype = next(self.base.parameters()).dtype
        self.head = nn.Linear(d_hidden, d_substrate, dtype=head_dtype)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.d_substrate = d_substrate

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns predicted h̄ of shape (B, d_substrate) in fp32 for stable MSE."""
        self.base.set_adapter(self.adapter_name)
        out = self.base(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, return_dict=True,
        )
        last_hidden = out.hidden_states[-1]                             # (B, T, d_hidden)
        if attention_mask is not None:
            # Take the last *non-padded* token per row.
            seq_lens = attention_mask.sum(dim=1) - 1                    # (B,)
            idx = seq_lens.view(-1, 1, 1).expand(-1, 1, last_hidden.shape[-1])
            last_token = last_hidden.gather(dim=1, index=idx).squeeze(1)
        else:
            last_token = last_hidden[:, -1, :]
        return self.head(last_token).float()                            # MSE in fp32

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
