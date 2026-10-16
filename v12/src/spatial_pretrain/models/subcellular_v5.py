"""Whole-cell scGPT anchor with a zero-initialized subcellular residual.

V5 keeps the frozen ordinary-scGPT CLS embedding as the exact step-zero
representation.  The V4 Local_pre -> cytoplasm Spatial -> Local_feedback
branch can learn a residual in that same space, while the sole reconstruction
head reads the final fused center-cell representation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .subcellular_v2 import patch_knn
from .subcellular_v4 import JointBinDecoder, SubcellularV4


class SubcellularV5(SubcellularV4):
    """Frozen whole-cell CLS plus a trainable V4 subcellular residual branch."""

    def __init__(self, asset_dir: str | Path, neighbors: int = 16,
                 checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = 'scgpt', backbone: nn.Module | None = None,
                 slots_per_compartment: int = 8, slot_dim: int = 64,
                 local_pre_layers: int = 1, spatial_layers: int = 2,
                 local_post_layers: int = 2, spatial_enabled: bool = True):
        if not isinstance(spatial_enabled, bool):
            raise ValueError('spatial_enabled must be boolean')
        super().__init__(asset_dir, neighbors=neighbors, checkpointing=checkpointing,
                         encoder_chunk_size=encoder_chunk_size,
                         backbone_kind=backbone_kind, backbone=backbone,
                         slots_per_compartment=slots_per_compartment, slot_dim=slot_dim,
                         local_pre_layers=local_pre_layers, spatial_layers=spatial_layers,
                         local_post_layers=local_post_layers)
        # No gate is used: a single zero-initialized projection both preserves
        # the exact pretrained anchor and receives a useful gradient at step 0.
        self.residual_projection = nn.Linear(2 * self.dim, self.backbone.dim, bias=False)
        nn.init.zeros_(self.residual_projection.weight)
        self.joint_decoder = JointBinDecoder(self.backbone.dim, self.backbone.dim, self.num_bins)
        self.spatial_enabled = spatial_enabled
        self.arch_config = dict(
            model_type='subcellular_whole_cls_residual_v5',
            backbone_kind=backbone_kind,
            backbone_dim=self.backbone.dim,
            slots_per_compartment=slots_per_compartment,
            slot_dim=slot_dim,
            branch_dim=2 * self.dim,
            residual_dim=self.backbone.dim,
            base_embedding='frozen_whole_cell_cls_l2',
            branch_order='local_pre_cytoplasm_spatial_local_feedback',
            fusion='base_plus_zero_initialized_linear_residual_no_post_norm',
            output_scope='center_cell_only',
            spatial_enabled=spatial_enabled,
            local_pre_layers=local_pre_layers,
            spatial_layers=spatial_layers,
            local_post_layers=local_post_layers,
            neighbors=neighbors,
            num_bins=self.num_bins,
            pooling_heads=self.arch_config['pooling_heads'],
            spatial_heads=self.arch_config['spatial_heads'],
        )
        # V5 always keeps its one shared pretrained backbone frozen.
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def set_backbone_trainable(self, full: bool):
        """V5 deliberately disallows fine-tuning the pretrained anchor."""
        if full:
            raise ValueError('V5 requires the ordinary scGPT backbone to remain frozen')
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def set_backbone_trainable_layers(self, count: int):
        if not isinstance(count, int) or isinstance(count, bool) or count != 0:
            raise ValueError('V5 requires zero trainable backbone layers')
        self.set_backbone_trainable(False)

    def _whole_cell_base(self, batch: Mapping[str, Tensor]) -> Tensor:
        """Encode only the center's masked whole-cell sequence as a frozen CLS."""
        try:
            genes = batch['whole_gene_ids']
            bins = batch['whole_input_bins']
            valid = batch['whole_token_valid_mask']
        except KeyError as error:
            raise KeyError(f'V5 batch is missing required center-only field {error.args[0]!r}') from error
        if genes.ndim != 2 or bins.shape != genes.shape or valid.shape != genes.shape:
            raise ValueError('whole_gene_ids/whole_input_bins/whole_token_valid_mask must share [B,L] shape')
        if genes.shape[0] < 1 or genes.shape[1] < 1:
            raise ValueError('Whole-cell input must contain at least one center and its CLS')
        packed_genes = batch.get('gene_ids')
        if packed_genes is None or packed_genes.ndim != 3 or packed_genes.shape[0] != genes.shape[0]:
            raise ValueError('Whole-cell and packed compartment inputs must have the same batch size')
        if valid.dtype != torch.bool:
            raise ValueError('whole_token_valid_mask must be boolean')
        if not bool(valid[:, 0].all()) or not bool(genes[:, 0].eq(self.cls_id).all()):
            raise ValueError('Each whole-cell center must retain a valid CLS at slot zero')
        # PAD is always excluded, even if an upstream validity mask is overly
        # permissive. Invalid payload values therefore cannot enter CLS attention.
        effective_valid = valid & genes.ne(self.pad_id)
        padding = ~effective_valid
        try:
            dtype = next(self.backbone.parameters()).dtype
        except StopIteration:
            dtype = bins.dtype if bins.is_floating_point() else torch.float32
        addition = torch.zeros((*genes.shape, self.backbone.dim), device=genes.device, dtype=dtype)
        with torch.no_grad():
            hidden = self.backbone(genes, bins, padding, addition)
            base = F.normalize(hidden[:, 0].float(), dim=-1, eps=1e-8)
        return base

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False,
                *, spatial: bool | None = None):
        """Return center-only embeddings and, unless requested otherwise, CE logits."""
        if spatial is None:
            spatial = self.spatial_enabled
        if not isinstance(spatial, bool):
            raise ValueError('spatial must be boolean')
        base = self._whole_cell_base(batch)
        initial, visible = self.encode_compartments(batch)
        local = initial
        for block in self.local_pre_blocks:
            local = block(local)

        graph = None
        if spatial:
            graph = patch_knn(batch['coords'], batch['node_indices'], visible[..., 1], self.neighbors)
        final = self._after_local(local, graph, spatial=spatial)
        local_states = self.normalized_states(local, batch['node_indices'])[:, 0]
        final_states = self.normalized_states(final, batch['node_indices'])[:, 0]
        branch_joint = final_states.flatten(-2)
        residual = self.residual_projection(branch_joint)
        # Deliberately no normalization after addition: zero residual must make
        # joint bitwise identical to the frozen base at initialization.
        joint = base + residual
        embeddings = dict(
            base=base,
            joint=joint,
            nucleus=final_states[:, 0],
            cytoplasm=final_states[:, 1],
            nucleus_local=local_states[:, 0],
            cytoplasm_local=local_states[:, 1],
        )
        result = dict(embeddings=embeddings, compartment_visible_mask=visible,
                      branch_joint=branch_joint, residual=residual)
        if return_embeddings:
            if spatial:
                intrinsic_states = self._after_local(local, None, spatial=False)
                intrinsic_states = self.normalized_states(
                    intrinsic_states, batch['node_indices'])[:, 0]
            else:
                intrinsic_states = final_states
            branch_intrinsic = intrinsic_states.flatten(-2)
            embeddings['intrinsic'] = base + self.residual_projection(branch_intrinsic)
            result['branch_intrinsic'] = branch_intrinsic
            return result

        query_compartments = batch['query_compartment_ids']
        if bool(((query_compartments < 0) | (query_compartments > 1)).any()):
            raise ValueError('Query compartments must be nuclear=0 or cytoplasm=1')
        query = (self.backbone.encoder(batch['query_gene_ids'])
                 + self.compartment_embedding(query_compartments))
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        context = joint[:, None, :].expand(-1, query.shape[1], -1)
        result['joint_logits'] = self.joint_decoder(query, context)
        return result
