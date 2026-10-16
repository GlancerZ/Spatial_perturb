"""Alternating subcellular/spatial attention with directed, bounded LR bias.

Route A adds a mixed subcellular/spatial residual to frozen ordinary-scGPT CLS.
Route B separately projects a self-only readout and the difference between
spatial and self-only readouts. Its context contribution is exactly zero when
messages are disabled or the center has no eligible incoming neighbors.

``lr_bias[b, i, j]`` is the visible-input-only score for sender j -> receiver i
in [0, 1]: weighted activity sum / (known pair weight + 4). The fixed evidence
shrinkage avoids dilution by the entire LR database. Global-prior coverage is
diagnostic and never multiplies this score. Neither branch reads targets.
"""
from __future__ import annotations

from math import isfinite, log
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .subcellular_v2 import LocalSpatialAttention, gather_neighbors, patch_knn
from .subcellular_v5 import SubcellularV5


class IntraSlotAttention(nn.Module):
    """Exchange the 2*K slots of each cell, without a premature FFN."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)

    def forward(self, states: Tensor) -> Tensor:
        shape = states.shape
        sequence = states.reshape(-1, shape[-3] * shape[-2], shape[-1])
        normalized = self.norm(sequence)
        message, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return (sequence + message).reshape(shape)


class SlotFeedForward(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, states: Tensor) -> Tensor:
        return states + self.layers(self.norm(states))


class LRSpatialAttention(LocalSpatialAttention):
    """Hard spatial eligibility plus QK, distance and directed LR soft bias."""

    def __init__(self, dim: int, heads: int, *, lr_enabled: bool = True,
                 lr_max_strength: float = 2., lr_initial_strength: float = .1):
        super().__init__(dim, heads)
        if not isinstance(lr_enabled, bool):
            raise ValueError('lr_enabled must be boolean')
        if (not isfinite(lr_max_strength) or not isfinite(lr_initial_strength)
                or not 0 < lr_initial_strength < lr_max_strength):
            raise ValueError('LR strengths must satisfy 0 < initial < max, both finite')
        ratio = lr_initial_strength / lr_max_strength
        self.lr_scale_logit = nn.Parameter(torch.full((heads,), log(ratio / (1 - ratio))),
                                          requires_grad=lr_enabled)
        self.lr_enabled = lr_enabled
        self.lr_max_strength = float(lr_max_strength)

    @property
    def lr_strength(self) -> Tensor:
        return self.lr_max_strength * self.lr_scale_logit.sigmoid()

    def forward(self, receiver: Tensor, senders: Tensor, graph,
                lr_bias: Tensor | None = None) -> Tensor:
        indices, valid, distance = graph
        batch, cells, dim = receiver.shape
        k = indices.shape[-1]
        q = self.query(self.norm(receiver)).reshape(batch * cells, self.heads, 1, dim // self.heads)
        normalized = self.norm(senders)
        keys = gather_neighbors(self.key(normalized), indices)
        values = gather_neighbors(self.value(normalized), indices)
        keys = keys.reshape(batch * cells, k, self.heads, dim // self.heads).transpose(1, 2)
        values = values.reshape(batch * cells, k, self.heads, dim // self.heads).transpose(1, 2)
        any_neighbor = valid.any(-1)
        safe = valid.clone()
        safe[..., 0] |= ~any_neighbor
        # Invalid distances cannot contaminate the dummy empty-neighborhood row.
        distance = distance.masked_fill(~valid, 0.)
        # Keep the additive bias in float32 under bf16 autocast: sparse LR
        # evidence can be tiny relative to a learned distance offset. Adding
        # both in bf16 would round away a legitimate LR perturbation early.
        bias = self.distance_bias(torch.log1p(distance).unsqueeze(-1).to(receiver.dtype)).float()
        if self.lr_enabled:
            if lr_bias is None:
                raise ValueError('LR-enabled spatial attention requires lr_bias')
            picked_lr = lr_bias.gather(-1, indices).masked_fill(~valid, 0.)
            bias = bias + picked_lr.unsqueeze(-1).float() * self.lr_strength.float()
        bias = bias.permute(0, 1, 3, 2).reshape(batch * cells, self.heads, 1, k)
        bias = bias.masked_fill(~safe.reshape(batch * cells, 1, 1, k), float('-inf'))
        # SDPA itself is autocast-eligible and would otherwise cast even the
        # float32 mask back to bf16. This short [1,K] attention therefore keeps
        # QK accumulation, additive biases and softmax explicitly in float32.
        with torch.autocast(device_type=receiver.device.type, enabled=False):
            message = F.scaled_dot_product_attention(q.float(), keys.float(), values.float(), attn_mask=bias)
        message = self.out(message.reshape(batch, cells, dim))
        return torch.where(any_neighbor.unsqueeze(-1), message, torch.zeros_like(message))


class AlternatingSlotBlock(nn.Module):
    """Intra -> cytoplasm-only spatial message -> token-wise FFN."""

    def __init__(self, slot_dim: int, slot_heads: int, cell_dim: int, spatial_heads: int,
                 *, lr_enabled: bool, lr_max_strength: float, lr_initial_strength: float):
        super().__init__()
        self.intra = IntraSlotAttention(slot_dim, slot_heads)
        self.spatial = LRSpatialAttention(
            cell_dim, spatial_heads, lr_enabled=lr_enabled,
            lr_max_strength=lr_max_strength, lr_initial_strength=lr_initial_strength)
        self.ffn = SlotFeedForward(slot_dim)

    def forward(self, states: Tensor, graph=None, lr_bias: Tensor | None = None) -> Tensor:
        states = self.intra(states)
        if graph is not None:
            cytoplasm = states[..., 1, :, :]
            flat = cytoplasm.flatten(-2)
            message = self.spatial(flat, flat, graph, lr_bias).reshape_as(cytoplasm)
            states = torch.stack((states[..., 0, :, :], cytoplasm + message), dim=-3)
        return self.ffn(states)


class SubcellularV6(SubcellularV5):
    """Shared frozen encoder and alternating slots, with A/B residual readouts.

    The internal attention/FFN paths intentionally use nonzero initialization.
    Zeroing both spatial messages and B's context projection would create a
    permanently dead context branch: neither would receive an opening gradient.
    Only the final residual projections are zero-initialized.
    """

    def __init__(self, asset_dir: str | Path, neighbors: int = 16,
                 checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = 'scgpt', backbone: nn.Module | None = None,
                 slots_per_compartment: int = 8, slot_dim: int = 64, num_blocks: int = 2,
                 spatial_enabled: bool = True, route: str = 'A', lr_enabled: bool = True,
                 lr_max_strength: float = 2., lr_initial_strength: float = .1,
                 max_neighbor_distance: float | None = None):
        if route not in ('A', 'B'):
            raise ValueError("route must be 'A' or 'B'")
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks < 1:
            raise ValueError('num_blocks must be a positive integer')
        if (max_neighbor_distance is not None
                and (not isfinite(max_neighbor_distance) or max_neighbor_distance < 0)):
            raise ValueError('max_neighbor_distance must be finite and nonnegative, or None')
        super().__init__(asset_dir, neighbors=neighbors, checkpointing=checkpointing,
                         encoder_chunk_size=encoder_chunk_size, backbone_kind=backbone_kind,
                         backbone=backbone, slots_per_compartment=slots_per_compartment,
                         slot_dim=slot_dim, local_pre_layers=0, spatial_layers=0,
                         local_post_layers=0, spatial_enabled=spatial_enabled)
        slot_heads = self.arch_config['pooling_heads']
        spatial_heads = self.arch_config['spatial_heads']
        self.blocks = nn.ModuleList([
            AlternatingSlotBlock(slot_dim, slot_heads, self.dim, spatial_heads,
                                 lr_enabled=lr_enabled, lr_max_strength=lr_max_strength,
                                 lr_initial_strength=lr_initial_strength)
            for _ in range(num_blocks)])
        self.final_intra = IntraSlotAttention(slot_dim, slot_heads)
        # Empty V5 module lists carry no parameters, but removing them avoids
        # advertising the obsolete Local_pre -> Spatial -> Local_feedback API.
        del self.local_pre_blocks, self.spatial_blocks, self.local_post_blocks
        self.route, self.lr_enabled = route, lr_enabled
        self.max_neighbor_distance = max_neighbor_distance
        if route == 'B':
            self.self_projection = self.residual_projection
            del self.residual_projection
            self.context_projection = nn.Linear(2 * self.dim, self.backbone.dim, bias=False)
            nn.init.zeros_(self.context_projection.weight)
        self.arch_config = dict(
            model_type='subcellular_alternating_lr_v6', route=route, backbone_kind=backbone_kind,
            backbone_dim=self.backbone.dim, slots_per_compartment=slots_per_compartment,
            slot_dim=slot_dim, branch_dim=2 * self.dim, residual_dim=self.backbone.dim,
            base_embedding='frozen_whole_cell_cls_l2', num_blocks=num_blocks,
            branch_order='repeat_intra_cytoplasm_spatial_ffn_then_final_intra',
            fusion=('base_plus_zero_initialized_mixed_residual_no_post_norm' if route == 'A'
                    else 'base_plus_zero_initialized_self_and_context_difference_no_post_norm'),
            self_context_shared_weights=True, internal_initialization='standard_nonzero',
            output_scope='center_cell_only', spatial_enabled=spatial_enabled,
            neighbors=neighbors, max_neighbor_distance=max_neighbor_distance,
            graph_scope='patch_knn_no_self_duplicate_or_pad_senders',
            lr_enabled=lr_enabled, lr_max_strength=float(lr_max_strength),
            lr_initial_strength=float(lr_initial_strength),
            lr_direction='receiver_i_sender_j', lr_bias_range=[0., 1.],
            lr_bias_reliability_shrinkage=4., lr_coverage_role='diagnostic_only_never_multiply',
            lr_scale='bounded_sigmoid_per_head', spatial_attention_dtype='float32',
            num_bins=self.num_bins, pooling_heads=slot_heads, spatial_heads=spatial_heads,
            # Checkpointing/chunk size are execution settings recorded in the
            # run config; evaluation can change them without changing weights.
            projection_parameters=sum(parameter.numel() for name, parameter in self.named_parameters()
                                      if name.startswith(('residual_projection.', 'self_projection.',
                                                          'context_projection.'))),
        )

    def set_backbone_trainable(self, full: bool):
        if full:
            raise ValueError('V6 requires the ordinary scGPT backbone to remain frozen')
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def set_backbone_trainable_layers(self, count: int):
        if not isinstance(count, int) or isinstance(count, bool) or count != 0:
            raise ValueError('V6 requires zero trainable backbone layers')
        self.set_backbone_trainable(False)

    def spatial_graph(self, batch: Mapping[str, Tensor], visible: Tensor):
        coords, ids = batch['coords'], batch['node_indices']
        if coords.ndim != 3 or coords.shape[:2] != ids.shape or coords.shape[-1] < 1:
            raise ValueError('coords must be [B,N,D] matching node_indices [B,N]')
        if not bool(torch.isfinite(coords[ids.ge(0)]).all()):
            raise ValueError('Valid cells require finite coordinates')
        # PAD coordinates may be arbitrary; they are never eligible graph nodes.
        coords = torch.where(ids.ge(0).unsqueeze(-1), coords, torch.zeros_like(coords))
        indices, valid, distance = patch_knn(coords, ids, visible[..., 1], self.neighbors)
        if self.max_neighbor_distance is not None:
            valid = valid & distance.le(self.max_neighbor_distance)
        return indices, valid, distance

    def _lr_inputs(self, batch: Mapping[str, Tensor]) -> tuple[Tensor | None, Tensor | None]:
        if not self.lr_enabled:
            return None, None
        shape = (*batch['node_indices'].shape, batch['node_indices'].shape[1])
        values = []
        for name in ('lr_bias', 'lr_coverage'):
            value = batch[name]
            if value.shape != shape or not value.is_floating_point():
                raise ValueError(f'{name} must be floating point [B,N,N] (receiver i, sender j)')
            if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
                raise ValueError(f'{name} must be finite and bounded in [0,1]')
            values.append(value)
        return values[0], values[1]

    def interact(self, initial: Tensor, graph=None, *, spatial: bool = True,
                 lr_bias: Tensor | None = None) -> Tensor:
        states = initial
        for block in self.blocks:
            states = block(states, graph if spatial else None, lr_bias)
        return self.final_intra(states)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False,
                *, spatial: bool | None = None):
        if spatial is None:
            spatial = self.spatial_enabled
        if not isinstance(spatial, bool):
            raise ValueError('spatial must be boolean')
        base = self._whole_cell_base(batch)
        initial, visible = self.encode_compartments(batch)
        graph = self.spatial_graph(batch, visible) if spatial else None
        lr_bias, coverage = self._lr_inputs(batch) if spatial else (None, None)
        final = self.interact(initial, graph, spatial=spatial, lr_bias=lr_bias)
        final_states = self.normalized_states(final, batch['node_indices'])[:, 0]
        branch_joint = final_states.flatten(-2)
        need_self = self.route == 'B' or return_embeddings
        self_states = None
        if need_self:
            self_only = self.interact(initial, spatial=False) if spatial else final
            self_states = self.normalized_states(self_only, batch['node_indices'])[:, 0]
            branch_self = self_states.flatten(-2)
        if self.route == 'A':
            residual = self.residual_projection(branch_joint)
            joint = base + residual
            intrinsic = base + self.residual_projection(branch_self) if need_self else None
        else:
            self_delta = self.self_projection(branch_self)
            context_features = branch_joint - branch_self
            if spatial:
                incoming = graph[1][:, 0].any(-1, keepdim=True)
                context_features = torch.where(incoming, context_features, torch.zeros_like(context_features))
            context_delta = self.context_projection(context_features)
            intrinsic = base + self_delta
            joint = intrinsic + context_delta
            residual = self_delta + context_delta
        # For ordinary training A avoids the extra self-only forward. Its local
        # diagnostic is the first Intra, which precedes all spatial messages.
        local_states = self.normalized_states(self.blocks[0].intra(initial), batch['node_indices'])[:, 0]
        embeddings = dict(base=base, joint=joint, nucleus=final_states[:, 0],
                          cytoplasm=final_states[:, 1], nucleus_local=local_states[:, 0],
                          cytoplasm_local=local_states[:, 1])
        result = dict(embeddings=embeddings, compartment_visible_mask=visible,
                      branch_joint=branch_joint, residual=residual)
        if need_self:
            embeddings['intrinsic'] = intrinsic
            result['branch_intrinsic'] = branch_self
        if self.route == 'B':
            embeddings.update(self=intrinsic, context_delta=context_delta)
            result.update(self_delta=self_delta, context_delta=context_delta)
        if graph is not None:
            result['spatial_neighbor_indices'], result['spatial_neighbor_mask'] = graph[:2]
            if coverage is not None:
                indices, valid, _ = graph
                weight = valid.to(coverage.dtype)
                denominator = weight.sum().clamp_min(1)
                result['lr_diagnostics'] = dict(
                    coverage_mean=(coverage.gather(-1, indices) * weight).sum().detach() / denominator,
                    bias_mean=(lr_bias.gather(-1, indices) * weight).sum().detach() / denominator,
                    nonzero_fraction=((lr_bias.gather(-1, indices) > 0) & valid).sum().detach() / denominator,
                    strength=torch.stack([block.spatial.lr_strength.detach() for block in self.blocks]),
                )
        if return_embeddings:
            return result
        query_compartments = batch['query_compartment_ids']
        if bool(((query_compartments < 0) | (query_compartments > 1)).any()):
            raise ValueError('Query compartments must be nuclear=0 or cytoplasm=1')
        query = self.backbone.encoder(batch['query_gene_ids']) + self.compartment_embedding(query_compartments)
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        context = joint[:, None, :].expand(-1, query.shape[1], -1)
        result['joint_logits'] = self.joint_decoder(query, context)
        return result
