"""Activation Verbalizer (AV).

AV reads a substrate residual `h_l` (384-d), projects it via P_AV to the AV
base's hidden size (Qwen3B: 2048), and injects it at every <ACT> position in
the prompt's input embeddings. The base then generates a 1-2 sentence
description as a normal chat-completion.

Frozen: AV base weights (4-bit Qwen).
Trainable: LoRA adapter "av" (rank 16 on q_proj+v_proj) + P_AV linear.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model


class AV(nn.Module):
    def __init__(
        self,
        base: nn.Module,
        act_token_id: int,
        d_substrate: int = 384,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
        adapter_name: str = "av",
    ):
        super().__init__()
        cfg = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=list(target_modules), task_type="CAUSAL_LM",
        )
        self.base = get_peft_model(base, cfg, adapter_name=adapter_name)
        self.adapter_name = adapter_name
        d_hidden = base.config.hidden_size
        # bnb 4-bit packs weights as torch.uint8; skip integer-typed params
        # so the projection ends up in a real floating-point dtype.
        proj_dtype = next(
            (p.dtype for p in base.parameters() if p.dtype.is_floating_point),
            torch.float16,
        )
        self.proj = nn.Linear(d_substrate, d_hidden, dtype=proj_dtype)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)
        self.act_id = int(act_token_id)
        self.d_substrate = d_substrate

    def _inject(self, input_ids: torch.Tensor, h_l: torch.Tensor) -> torch.Tensor:
        emb = self.base.get_input_embeddings()(input_ids)              # (B, T, d_hidden)
        # h_l comes from the dataloader as fp32; P_AV weights inherit the base
        # dtype (bf16 on Qwen 3B / T4). Cast input to weight dtype before linear.
        h_cast = h_l.to(self.proj.weight.dtype)
        inj = self.proj(h_cast).unsqueeze(1).to(emb.dtype)             # (B, 1, d_hidden)
        mask = (input_ids == self.act_id).unsqueeze(-1)                 # (B, T, 1)
        return torch.where(mask, inj.expand_as(emb), emb)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h_l: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        self.base.set_adapter(self.adapter_name)
        emb = self._inject(input_ids, h_l)
        return self.base(inputs_embeds=emb, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        h_l: torch.Tensor,
        max_new_tokens: int = 80,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy decode with <ACT> injection on the prompt forward.

        Returns: full token ids tensor (B, T_prompt + N_new), padded with
        pad_token_id for batch entries that finished early.
        """
        self.base.set_adapter(self.adapter_name)
        device = input_ids.device
        B = input_ids.shape[0]
        if pad_token_id is None:
            pad_token_id = eos_token_id if eos_token_id is not None else 0

        # Step 0: prompt forward via inputs_embeds so <ACT> injection lands.
        emb = self._inject(input_ids, h_l)
        out = self.base(
            inputs_embeds=emb, attention_mask=attention_mask,
            use_cache=True, return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)            # (B, 1)
        generated = [next_id]
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        if eos_token_id is not None:
            finished |= (next_id.squeeze(-1) == eos_token_id)

        cur_attention = torch.cat(
            [attention_mask, torch.ones((B, 1), dtype=attention_mask.dtype, device=device)],
            dim=-1,
        )

        for _ in range(max_new_tokens - 1):
            if finished.all():
                break
            out = self.base(
                input_ids=next_id, attention_mask=cur_attention,
                past_key_values=past, use_cache=True, return_dict=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = logits.argmax(dim=-1, keepdim=True)
            # Force pad on already-finished sequences.
            next_id = torch.where(finished.unsqueeze(-1), torch.full_like(next_id, pad_token_id), next_id)
            generated.append(next_id)
            if eos_token_id is not None:
                finished |= (next_id.squeeze(-1) == eos_token_id)
            cur_attention = torch.cat(
                [cur_attention, torch.ones((B, 1), dtype=cur_attention.dtype, device=device)],
                dim=-1,
            )

        new_ids = torch.cat(generated, dim=1)
        return torch.cat([input_ids, new_ids], dim=1)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
