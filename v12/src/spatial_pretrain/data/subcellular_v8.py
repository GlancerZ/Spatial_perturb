"""V8 pilot data: V7 tasks with a selectable quantile-tie policy.

``binning.tie_policy: left`` reproduces V7 exactly (deterministic NumPy
``digitize``). ``random`` reproduces V6 exactly: randomized tie spreading for
compartment inputs, whole-cell inputs and full-panel targets. Masking, query
sampling, token subsampling and LR evidence are shared by both policies.
"""
from __future__ import annotations

from .subcellular_v3 import encode_context_v3
from .subcellular_v5 import encode_whole_cell_v5
from .subcellular_v6 import V6Dataset
from .subcellular_v7 import (
    V7Dataset,
    collate_v7,
    encode_compartment_context_v7,
    encode_whole_cell_v7,
)

TIE_POLICIES = ("left", "random")


def tie_policy(config) -> str:
    policy = config.get("binning", {}).get("tie_policy")
    if policy not in TIE_POLICIES:
        raise ValueError(f"binning.tie_policy must be one of {TIE_POLICIES}")
    return policy


def compartment_encoder(policy: str):
    """Evaluation-time N/C packer with the same signature for both policies."""
    return encode_compartment_context_v7 if policy == "left" else encode_context_v3


def whole_cell_encoder(policy: str):
    return encode_whole_cell_v7 if policy == "left" else encode_whole_cell_v5


class V8Dataset(V7Dataset):
    def __init__(self, config: dict, split="train"):
        super().__init__(config, split=split)
        self.tie_policy = tie_policy(config)

    def __getitem__(self, index):
        if self.tie_policy == "random":
            return V6Dataset.__getitem__(self, index)
        return V7Dataset.__getitem__(self, index)


def collate_v8(batch, *, pad_token_id=None):
    return collate_v7(batch, pad_token_id=pad_token_id)
