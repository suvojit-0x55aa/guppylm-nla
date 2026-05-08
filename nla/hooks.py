"""Forward hooks on every Block to capture post-residual outputs."""

from typing import Dict, List, Tuple

import torch
from torch.utils.hooks import RemovableHandle


def register_block_hooks(model) -> Tuple[Dict[int, List[torch.Tensor]], List[RemovableHandle]]:
    """Hook every Block.forward → captures detached output (B, T, d_model).

    Returns (storage, handles).
    - storage[i] is a list; each forward appends one tensor. Caller must clear
      between samples. Use .clear() per layer or replace the list.
    - handles must be .remove()'d when extraction is done.
    """
    storage: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(model.blocks))}

    def _make(i):
        def _hook(_module, _inputs, output):
            storage[i].append(output.detach())
        return _hook

    handles = [block.register_forward_hook(_make(i)) for i, block in enumerate(model.blocks)]
    return storage, handles
