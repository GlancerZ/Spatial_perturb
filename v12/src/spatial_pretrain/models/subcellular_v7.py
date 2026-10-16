"""V7: deterministic-bin V6 routes with last-layer scGPT fine-tuning."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .subcellular_v6 import SubcellularV6


class SubcellularV7(SubcellularV6):
    """Keep the V6 A/B architecture while fine-tuning a backbone suffix."""

    def __init__(self, asset_dir: str | Path, neighbors: int = 16,
                 checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = "scgpt", backbone: nn.Module | None = None,
                 slots_per_compartment: int = 8, slot_dim: int = 64,
                 num_blocks: int = 2, spatial_enabled: bool = True,
                 route: str = "A", lr_enabled: bool = True,
                 lr_max_strength: float = 2., lr_initial_strength: float = .1,
                 max_neighbor_distance: float | None = None):
        self._backbone_trainable_layers = 0
        super().__init__(
            asset_dir, neighbors=neighbors, checkpointing=checkpointing,
            encoder_chunk_size=encoder_chunk_size, backbone_kind=backbone_kind,
            backbone=backbone, slots_per_compartment=slots_per_compartment,
            slot_dim=slot_dim, num_blocks=num_blocks,
            spatial_enabled=spatial_enabled, route=route, lr_enabled=lr_enabled,
            lr_max_strength=lr_max_strength,
            lr_initial_strength=lr_initial_strength,
            max_neighbor_distance=max_neighbor_distance)
        self.arch_config.update(
            model_type="subcellular_deterministic_lr_routes_v7",
            base_embedding="fine_tunable_whole_cell_cls_l2",
            input_binning="deterministic_left_per_cell_51_bins",
            backbone_finetune="last_n_transformer_layers_configured_by_training",
            backbone_trainable_layers=0,
        )

    @property
    def backbone_trainable_layers(self) -> int:
        return self._backbone_trainable_layers

    def set_backbone_trainable(self, full: bool):
        """V7 uses an explicit layer count and never silently unfreezes all."""
        if full:
            raise ValueError("Use set_backbone_trainable_layers for V7")
        self._backbone_trainable_layers = 0
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        if hasattr(self, "arch_config"):
            self.arch_config["backbone_trainable_layers"] = 0

    def _apply_backbone_modes(self, mode: bool):
        count = self._backbone_trainable_layers
        if not count or not mode:
            self.backbone.eval()
            return
        # Keep the backbone root in training mode so ScGPTBackbone activates
        # non-reentrant gradient checkpointing. Frozen modules remain eval.
        self.backbone.train(True)
        self.backbone.encoder.eval()
        self.backbone.value_encoder.eval()
        layers = self.backbone.transformer_encoder.layers
        for layer in layers[:-count]:
            layer.eval()
        for layer in layers[-count:]:
            layer.train(True)
        if self.backbone.transformer_encoder.norm is not None:
            self.backbone.transformer_encoder.norm.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_backbone_modes(bool(mode))
        return self

    def set_backbone_trainable_layers(self, count: int):
        layers = self.backbone.transformer_encoder.layers
        if (not isinstance(count, int) or isinstance(count, bool)
                or not 0 <= count <= len(layers)):
            raise ValueError(f"count must be an integer between 0 and {len(layers)}")
        self.backbone.requires_grad_(False)
        if count:
            for layer in layers[-count:]:
                layer.requires_grad_(True)
        self._backbone_trainable_layers = count
        self.arch_config["backbone_trainable_layers"] = count
        self._apply_backbone_modes(self.training)

    def _whole_cell_base(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Encode the visible center CLS with gradients for the trainable suffix."""
        try:
            genes = batch["whole_gene_ids"]
            bins = batch["whole_input_bins"]
            valid = batch["whole_token_valid_mask"]
        except KeyError as error:
            raise KeyError(
                f"V7 batch is missing required center-only field {error.args[0]!r}") from error
        if genes.ndim != 2 or bins.shape != genes.shape or valid.shape != genes.shape:
            raise ValueError("whole-cell gene/bin/valid arrays must share [B,L] shape")
        if genes.shape[0] < 1 or genes.shape[1] < 1:
            raise ValueError("Whole-cell input must retain at least CLS")
        packed = batch.get("gene_ids")
        if packed is None or packed.ndim != 3 or packed.shape[0] != genes.shape[0]:
            raise ValueError("Whole-cell and compartment inputs need the same batch size")
        if valid.dtype != torch.bool:
            raise ValueError("whole_token_valid_mask must be boolean")
        if not bool(valid[:, 0].all()) or not bool(genes[:, 0].eq(self.cls_id).all()):
            raise ValueError("Each whole-cell center must retain a valid CLS")
        effective_valid = valid & genes.ne(self.pad_id)
        padding = ~effective_valid
        dtype = next(self.backbone.parameters()).dtype
        addition = torch.zeros(
            (*genes.shape, self.backbone.dim), device=genes.device, dtype=dtype)
        if self._backbone_trainable_layers:
            hidden = self.backbone(genes, bins, padding, addition)
        else:
            with torch.no_grad():
                hidden = self.backbone(genes, bins, padding, addition)
        return F.normalize(hidden[:, 0].float(), dim=-1, eps=1e-8)
