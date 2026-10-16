"""V8 pilot: V7 routes with an optional soft norm cap on the CLS residual.

V6/V7 add an unconstrained residual to a unit-norm CLS; at the end of V7 the
residual/base norm ratio was ~1.15-1.18 and shrinking it post hoc improved
both probes. ``residual_cap`` bounds ``||r||`` smoothly below ``cap*||base||``
with ``r * cap*||b|| * tanh(||r||/(cap*||b||)) / ||r||``. The scale is exactly
one at ``r = 0``, so step zero still equals the ordinary CLS and the
zero-initialized projection keeps its opening gradient.
"""
from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .subcellular_v7 import SubcellularV7


class SubcellularV8(SubcellularV7):
    def __init__(self, asset_dir: str | Path, neighbors: int = 16,
                 checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = "scgpt", backbone: nn.Module | None = None,
                 slots_per_compartment: int = 8, slot_dim: int = 64,
                 num_blocks: int = 2, spatial_enabled: bool = True,
                 route: str = "A", lr_enabled: bool = True,
                 lr_max_strength: float = 2., lr_initial_strength: float = .1,
                 max_neighbor_distance: float | None = None,
                 residual_cap: float | None = None):
        if residual_cap is not None and (isinstance(residual_cap, bool)
                                         or not isfinite(residual_cap) or residual_cap <= 0):
            raise ValueError("residual_cap must be a positive finite number or None")
        super().__init__(
            asset_dir, neighbors=neighbors, checkpointing=checkpointing,
            encoder_chunk_size=encoder_chunk_size, backbone_kind=backbone_kind,
            backbone=backbone, slots_per_compartment=slots_per_compartment,
            slot_dim=slot_dim, num_blocks=num_blocks, spatial_enabled=spatial_enabled,
            route=route, lr_enabled=lr_enabled, lr_max_strength=lr_max_strength,
            lr_initial_strength=lr_initial_strength,
            max_neighbor_distance=max_neighbor_distance)
        self.residual_cap = None if residual_cap is None else float(residual_cap)
        self.arch_config.update(
            model_type="subcellular_capped_routes_v8",
            residual_cap=self.residual_cap,
            fusion=("base_plus_residual" if residual_cap is None
                    else "base_plus_tanh_soft_capped_residual"))

    def cap(self, residual: Tensor, base: Tensor) -> Tensor:
        if self.residual_cap is None:
            return residual
        residual = residual.float()
        limit = self.residual_cap * base.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        norm = residual.norm(dim=-1, keepdim=True)
        safe = norm.clamp_min(1e-12)
        scale = torch.where(norm > 1e-12, limit * torch.tanh(safe / limit) / safe,
                            torch.ones_like(norm))
        return residual * scale

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False,
                *, spatial: bool | None = None):
        if spatial is None:
            spatial = self.spatial_enabled
        if not isinstance(spatial, bool):
            raise ValueError("spatial must be boolean")
        base = self._whole_cell_base(batch)
        initial, visible = self.encode_compartments(batch)
        graph = self.spatial_graph(batch, visible) if spatial else None
        lr_bias, coverage = self._lr_inputs(batch) if spatial else (None, None)
        final = self.interact(initial, graph, spatial=spatial, lr_bias=lr_bias)
        final_states = self.normalized_states(final, batch["node_indices"])[:, 0]
        branch_joint = final_states.flatten(-2)
        need_self = self.route == "B" or return_embeddings
        if need_self:
            self_only = self.interact(initial, spatial=False) if spatial else final
            self_states = self.normalized_states(self_only, batch["node_indices"])[:, 0]
            branch_self = self_states.flatten(-2)
        if self.route == "A":
            residual = self.cap(self.residual_projection(branch_joint), base)
            joint = base + residual
            intrinsic = (base + self.cap(self.residual_projection(branch_self), base)
                         if need_self else None)
        else:
            raw_self = self.self_projection(branch_self)
            context_features = branch_joint - branch_self
            if spatial:
                incoming = graph[1][:, 0].any(-1, keepdim=True)
                context_features = torch.where(incoming, context_features,
                                               torch.zeros_like(context_features))
            residual = self.cap(raw_self + self.context_projection(context_features), base)
            self_delta = self.cap(raw_self, base)
            intrinsic = base + self_delta
            joint = base + residual
            context_delta = residual - self_delta
        local_states = self.normalized_states(self.blocks[0].intra(initial),
                                              batch["node_indices"])[:, 0]
        embeddings = dict(base=base, joint=joint, nucleus=final_states[:, 0],
                          cytoplasm=final_states[:, 1], nucleus_local=local_states[:, 0],
                          cytoplasm_local=local_states[:, 1])
        result = dict(embeddings=embeddings, compartment_visible_mask=visible,
                      branch_joint=branch_joint, residual=residual)
        if need_self:
            embeddings["intrinsic"] = intrinsic
            result["branch_intrinsic"] = branch_self
        if self.route == "B":
            embeddings.update(self=intrinsic, context_delta=context_delta)
            result.update(self_delta=self_delta, context_delta=context_delta)
        if graph is not None:
            result["spatial_neighbor_indices"], result["spatial_neighbor_mask"] = graph[:2]
            if coverage is not None:
                indices, valid, _ = graph
                weight = valid.to(coverage.dtype)
                denominator = weight.sum().clamp_min(1)
                result["lr_diagnostics"] = dict(
                    coverage_mean=(coverage.gather(-1, indices) * weight).sum().detach() / denominator,
                    bias_mean=(lr_bias.gather(-1, indices) * weight).sum().detach() / denominator,
                    nonzero_fraction=((lr_bias.gather(-1, indices) > 0) & valid).sum().detach() / denominator,
                    strength=torch.stack([block.spatial.lr_strength.detach() for block in self.blocks]),
                )
        if return_embeddings:
            return result
        query_compartments = batch["query_compartment_ids"]
        if bool(((query_compartments < 0) | (query_compartments > 1)).any()):
            raise ValueError("Query compartments must be nuclear=0 or cytoplasm=1")
        query = self.backbone.encoder(batch["query_gene_ids"]) + self.compartment_embedding(query_compartments)
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        context = joint[:, None, :].expand(-1, query.shape[1], -1)
        result["joint_logits"] = self.joint_decoder(query, context)
        return result
