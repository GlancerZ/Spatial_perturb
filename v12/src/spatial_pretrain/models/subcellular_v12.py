"""V12: complementary nuclear/cytoplasmic counts with a spatial residual.

The two compartments retain independent ordinary-scGPT CLS readouts. A zero-
initialized token adapter exposes visible raw counts without perturbing the
pretrained representation at step zero. Spatial has no center expression input;
the separate internal environment readout may use visible center receptor
evidence. One shared joint count distribution supervises all available views.
This observation model does not claim to identify time or RNA velocity.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .subcellular_v11 import SubcellularV11, LRNeighborEncoder, _masked_softmax, _unit


MODES = ("nucleus", "cytoplasm", "local", "spatial", "fused")
MODELS = MODES  # Explicit compatibility alias for experiment orchestration.
READOUTS = (*MODES, "concat")
ARCHITECTURE = "subcellular_compartment_counts_v12"
MODEL_VERSION = 12


class LRNeighborEncoderV12(LRNeighborEncoder):
    """Keep pure spatial readout and optionally condition its twin on center LR.

    Graph edges are receiver i, sender j. center_lr_bias[k] instead represents
    sender neighbor k -> receiver center, from visible center receptors only.
    Coverage is a diagnostic, not a hidden target or a multiplicative gate.
    """

    def forward(self, neighbors, present, relative_coords, lr_bias=None, center_lr_bias=None):
        size, count, _ = neighbors.shape
        if not count:
            zero = sum(p.reshape(-1)[0] * 0. for p in self.parameters() if p.requires_grad)
            empty = neighbors.new_zeros((size, self.dim)) + zero
            return (empty, empty, neighbors.new_zeros((size, self.heads, 0, 0)),
                    neighbors.new_zeros((size, self.heads, 0)),
                    neighbors.new_zeros((size, self.heads, 0)),
                    torch.zeros((size, 0, 0), dtype=torch.bool, device=neighbors.device))
        eligible, offsets = self.graph(present, relative_coords)
        features = self.norm(neighbors)
        q, k, v = [layer(features).reshape(size, count, self.heads, self.head_dim)
                   for layer in (self.query, self.key, self.value)]
        with torch.autocast(device_type=neighbors.device.type, enabled=False):
            logits = torch.einsum("bihd,bjhd->bhij", q.float(), k.float()) * self.head_dim ** -.5
        logits = logits + self.edge_position(self._coordinates(offsets)).float().permute(0, 3, 1, 2)
        if self.lr_enabled:
            if lr_bias is None or lr_bias.shape != (size, count, count):
                raise ValueError("LR requires neighbor_lr_bias [B,K,K], receiver by sender")
            if not bool(torch.isfinite(lr_bias[eligible]).all()):
                raise ValueError("Eligible graph LR evidence must be finite")
            evidence = torch.where(eligible, lr_bias.float(), 0.).clamp(0., 1.)
            logits = logits + evidence[:, None] * self.lr_strength[None, :, None, None].float()
        attention = _masked_softmax(logits, eligible[:, None])
        messages = torch.einsum("bhij,bjhd->bihd", attention.to(v.dtype), v).reshape(size, count, self.dim)
        states = neighbors + self.output(messages) * eligible.any(-1, keepdim=True)
        states = (states + self.ffn(states)) * present[..., None]
        normalized = self.readout_norm(states)
        keys, values = [layer(normalized).reshape(size, count, self.heads, self.head_dim)
                        for layer in (self.readout_key, self.readout_value)]
        clean_coords = torch.where(present[..., None], relative_coords.float(), 0.)
        with torch.autocast(device_type=neighbors.device.type, enabled=False):
            pure_logits = torch.einsum("hd,bkhd->bhk", self.readout_query.float(), keys.float())
        pure_logits = pure_logits * self.head_dim ** -.5
        pure_logits = pure_logits + self.center_position(self._coordinates(clean_coords)).float().permute(0, 2, 1)
        condition_logits = pure_logits
        if self.lr_enabled and center_lr_bias is not None:
            if center_lr_bias.shape != present.shape or not bool(torch.isfinite(center_lr_bias[present]).all()):
                raise ValueError("center_lr_bias must be finite [B,K] on present neighbors")
            evidence = torch.where(present, center_lr_bias.float(), 0.).clamp(0., 1.)
            condition_logits = pure_logits + evidence[:, None] * self.lr_strength[None, :, None].float()
        pure_attention = _masked_softmax(pure_logits, present[:, None])
        condition_attention = _masked_softmax(condition_logits, present[:, None])

        def pool(weights):
            pooled = torch.einsum("bhk,bkhd->bhd", weights.to(values.dtype), values).reshape(size, self.dim)
            return (self.readout_output(pooled) * present.any(-1, keepdim=True)).float()

        return pool(pure_attention), pool(condition_attention), attention, pure_attention, condition_attention, eligible


class JointCountDecoder(nn.Module):
    """NB total and beta-binomial nuclear allocation; one normalized joint PMF."""

    def __init__(self, dim):
        super().__init__()
        self.query_norm, self.state_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.hidden = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU())
        self.output = nn.Linear(dim, 4)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, query, state):
        raw = self.output(self.hidden(torch.cat((self.query_norm(query), self.state_norm(state)), -1))).float()
        # Bounded transforms keep lgamma stable under mixed precision. Limits
        # describe count model parameters, never clip or rescale target counts.
        return dict(mu_total=(F.softplus(raw[..., 0]) + 1e-4).clamp_max(1e7),
                    theta_total=(F.softplus(raw[..., 1]) + 1e-3).clamp_max(1e5),
                    p_nucleus=1e-4 + (1 - 2e-4) * raw[..., 2].sigmoid(),
                    kappa_allocation=(F.softplus(raw[..., 3]) + 2e-3).clamp_max(1e5))


class SubcellularV12(SubcellularV11):
    def __init__(self, asset_dir: str | Path, *, count_adapter_hidden: int = 32,
                 fusion_initial_strength: float = .05, fusion_max_strength: float = .25,
                 backbone_trainable_layers: int = 2, **kwargs):
        if isinstance(count_adapter_hidden, bool) or not isinstance(count_adapter_hidden, int) or count_adapter_hidden < 1:
            raise ValueError("count_adapter_hidden must be a positive integer")
        if not (math.isfinite(fusion_initial_strength) and math.isfinite(fusion_max_strength)
                and 0 < fusion_initial_strength < fusion_max_strength <= 1):
            raise ValueError("Fusion strengths require 0 < initial < max <= 1")
        super().__init__(asset_dir, backbone_trainable_layers=backbone_trainable_layers, **kwargs)
        # Reuse V11's verified backbone/presence rules, replacing its entire
        # fusion/decoder stack. No disconnected V11 parameters remain registered.
        del self.fusion, self.fusion_nucleus_norm, self.fusion_spatial_norm, self.joint_decoder
        inherited_spatial_state = self.spatial_encoder.state_dict()
        self.spatial_encoder = LRNeighborEncoderV12(
            self.dim, self.arch_config["spatial_heads"],
            **{key: self.arch_config[key] for key in (
                "coordinate_scale", "spatial_radius", "graph_neighbors", "lr_enabled",
                "lr_max_strength", "lr_initial_strength", "dropout")})
        self.spatial_encoder.load_state_dict(inherited_spatial_state)
        self.raw_count_adapter = nn.Sequential(nn.Linear(1, count_adapter_hidden), nn.GELU(),
                                               nn.Linear(count_adapter_hidden, self.dim))
        nn.init.zeros_(self.raw_count_adapter[-1].weight)
        nn.init.zeros_(self.raw_count_adapter[-1].bias)
        self.local_nucleus_norm, self.local_cytoplasm_norm = nn.LayerNorm(self.dim), nn.LayerNorm(self.dim)
        self.local_gate = nn.Linear(2 * self.dim + 4, 1)
        nn.init.zeros_(self.local_gate.weight)
        nn.init.zeros_(self.local_gate.bias)
        self.fusion_local_norm, self.fusion_environment_norm = nn.LayerNorm(self.dim), nn.LayerNorm(self.dim)
        self.fusion_gate = nn.Linear(2 * self.dim + 4, 1)
        nn.init.zeros_(self.fusion_gate.weight)
        ratio = fusion_initial_strength / fusion_max_strength
        nn.init.constant_(self.fusion_gate.bias, math.log(ratio / (1 - ratio)))
        self.environment_residual = nn.Linear(self.dim, self.dim)
        self.fusion_max_strength = float(fusion_max_strength)
        self.joint_decoder = JointCountDecoder(self.dim)
        self.arch_config.update(
            model_type=ARCHITECTURE, model_version=MODEL_VERSION, readouts=list(READOUTS),
            decoder="shared_NB_total_times_beta_binomial_nuclear_allocation",
            count_adapter="visible_log1p_raw_counts_zero_initialized_token_residual",
            neighbor_count_adapter="used_only_if_aligned_neighbor_log_counts_are_provided",
            count_adapter_hidden=count_adapter_hidden, nucleus_view="nucleus_only_cls",
            cytoplasm_view="cytoplasm_only_cls", local="gated_N_C_average_missing_view_fallback",
            spatial_view="neighbors_only_graph_and_center_position_readout_no_center_RNA",
            internal_environment_view="spatial_readout_conditioned_on_visible_center_receptor_LR",
            fusion="local_plus_gated_norm_bounded_environment_residual_missing_view_fallback",
            fusion_initial_strength=float(fusion_initial_strength), fusion_max_strength=float(fusion_max_strength),
            concat="concat(unit_nucleus,unit_cytoplasm,unit_pure_spatial)",
            center_statistics="visible_only_log1p_N_sum_C_sum_N_positive_C_positive",
            objective="one_joint_count_likelihood_no_teacher_or_latent_alignment",
            direction_output="not_identified_by_observation_model")

    def _count_cls(self, genes, bins, visible, log_counts):
        if log_counts is None:
            # A missing count channel is not a measured log-count of zero.
            # In particular, neighbors using only bins must not acquire the
            # adapter's learned bias as a fabricated count observation.
            addition = bins.new_zeros((*genes.shape, self.dim))
        else:
            count_visible = visible & genes.ne(self.cls_id) & genes.ne(self.pad_id)
            addition = self.raw_count_adapter(log_counts[..., None]) * count_visible[..., None]
        return self.backbone(genes, bins, ~visible, addition)[:, 0].float()

    def encode_sequences(self, genes, bins, valid, log_counts=None):
        if genes.ndim != 2 or genes.shape != bins.shape or genes.shape != valid.shape or genes.shape[1] < 1:
            raise ValueError("Tokens require aligned [N,L] genes/bins/validity with L >= 1")
        if not len(genes):
            return bins.new_empty((0, self.dim), dtype=torch.float32)
        if not bool((genes[:, 0].eq(self.cls_id) & valid[:, 0].bool()).all()):
            raise ValueError("Every present sequence must start with valid CLS")
        visible = valid.bool() & genes.ne(self.pad_id)
        if not bool(torch.isfinite(bins[visible]).all()):
            raise ValueError("Visible bins must be finite")
        clean_genes = torch.where(visible, genes, self.pad_id)
        clean_bins = torch.where(visible, bins, self.pad_value).float()
        if log_counts is None:
            clean_counts = None
        else:
            if log_counts.shape != genes.shape:
                raise ValueError("log_counts must align to tokens")
            if not bool(torch.isfinite(log_counts[visible]).all()) or bool(log_counts[visible].lt(0).any()):
                raise ValueError("Visible log_counts must be finite nonnegative log1p raw counts")
            if bool(log_counts[visible & genes.eq(self.cls_id)].ne(0).any()):
                raise ValueError("CLS log_counts must be zero")
            clean_counts = torch.where(visible & genes.ne(self.cls_id), log_counts, 0.).float()
        outputs = []
        for start in range(0, len(genes), self.encoder_chunk_size):
            part = slice(start, start + self.encoder_chunk_size)
            args = (clean_genes[part], clean_bins[part], visible[part],
                    None if clean_counts is None else clean_counts[part])
            if self.checkpointing and self.training and torch.is_grad_enabled():
                outputs.append(checkpoint(self._count_cls, *args, use_reentrant=False))
            else:
                outputs.append(self._count_cls(*args))
        return torch.cat(outputs)

    def _v12_presence(self, batch):
        nucleus, present = super()._input_presence(batch)
        genes, valid = batch["cytoplasm_gene_ids"], batch["cytoplasm_valid"]
        if genes.ndim != 2 or genes.shape[0] != len(nucleus) or valid.shape != genes.shape:
            raise ValueError("Cytoplasm gene IDs and validity must align to [B,L]")
        cytoplasm = batch.get("cytoplasm_present", valid[:, 1:].any(-1)).bool()
        if cytoplasm.shape != nucleus.shape:
            raise ValueError("cytoplasm_present must have [B] shape")
        cytoplasm = cytoplasm & (valid.bool() & genes.ne(self.pad_id) & genes.ne(self.cls_id)).any(-1)
        return nucleus, cytoplasm, present

    def mode_availability(self, batch):
        nucleus, cytoplasm, present = self._v12_presence(batch)
        local, spatial = nucleus | cytoplasm, present.any(-1)
        return dict(nucleus=nucleus, cytoplasm=cytoplasm, local=local, spatial=spatial,
                    fused=local | spatial)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings=False):
        if any(key in batch for key in ("teacher_features", "nucleus_features", "neighbor_features")):
            raise ValueError("V12 encodes tokens; feature-cache/teacher inputs are unsupported")
        genes, bins, valid = [batch[f"neighbor_{name}"] for name in ("gene_ids", "bins", "valid")]
        if genes.ndim != 3 or genes.shape != bins.shape or genes.shape != valid.shape:
            raise ValueError("Neighbor tokens must share [B,K,L]")
        size, count, _ = genes.shape
        nucleus_available, cytoplasm_available, present = self._v12_presence(batch)
        if present.shape != (size, count):
            raise ValueError("Neighbor presence must match [B,K]")
        encoded = {}
        for name, available in (("nucleus", nucleus_available), ("cytoplasm", cytoplasm_available)):
            part = [batch[f"{name}_{suffix}"] for suffix in ("gene_ids", "bins", "valid", "log_counts")]
            if any(value.shape != part[0].shape for value in part) or part[0].shape[0] != size:
                raise ValueError(f"{name} inputs must share [B,L]")
            features = bins.new_zeros((size, self.dim), dtype=torch.float32)
            if bool(available.any()):
                features[available] = self.encode_sequences(*(value[available] for value in part))
            encoded[name] = features
        h_neighbors = bins.new_zeros((size, count, self.dim), dtype=torch.float32)
        neighbor_counts = batch.get("neighbor_log_counts")
        if neighbor_counts is not None and neighbor_counts.shape != genes.shape:
            raise ValueError("neighbor_log_counts must align to neighbor tokens")
        if bool(present.any()):
            h_neighbors[present] = self.encode_sequences(genes[present], bins[present], valid[present],
                None if neighbor_counts is None else neighbor_counts[present])
        # A cheap zero graph ensures DDP has defined zero gradients even when a
        # rank has only empty/masked views. All learned paths remain registered.
        zero = sum(p.reshape(-1)[0] * 0. for p in self.parameters() if p.requires_grad)
        h_n, h_c = encoded["nucleus"] + zero, encoded["cytoplasm"] + zero
        h_s, h_e, graph_attention, read_attention, condition_attention, graph = self.spatial_encoder(
            h_neighbors, present, batch["relative_coords"], batch.get("neighbor_lr_bias"), batch.get("center_lr_bias"))
        spatial_available, local_available = present.any(-1), nucleus_available | cytoplasm_available
        stats = batch["center_statistics"]
        if stats.shape != (size, 4) or not bool(torch.isfinite(stats).all()) or bool(stats.lt(0).any()):
            raise ValueError("center_statistics requires finite nonnegative visible-only [B,4]")
        # Missing compartments cannot leak their aggregate counts through gates.
        stat_present = torch.stack((nucleus_available, cytoplasm_available, nucleus_available, cytoplasm_available), -1)
        stats = torch.where(stat_present, stats.float(), 0.)
        local_weight = self.local_gate(torch.cat((self.local_nucleus_norm(h_n), self.local_cytoplasm_norm(h_c), stats), -1)).sigmoid()
        h_l = local_weight * h_n + (1 - local_weight) * h_c
        h_l = torch.where((nucleus_available & cytoplasm_available)[:, None], h_l,
                          torch.where(nucleus_available[:, None], h_n, h_c))
        gate = self.fusion_max_strength * self.fusion_gate(torch.cat((
            self.fusion_local_norm(h_l), self.fusion_environment_norm(h_e), stats), -1)).sigmoid()
        residual = _unit(self.environment_residual(self.fusion_environment_norm(h_e)))
        residual = residual * h_l.float().norm(dim=-1, keepdim=True).clamp_min(1e-8)
        h_f = h_l + gate * residual
        paired = local_available & spatial_available
        h_f = torch.where(paired[:, None], h_f, torch.where(local_available[:, None], h_l, h_s))
        embeddings = dict(nucleus=h_n, cytoplasm=h_c, local=h_l, spatial=h_s, fused=h_f,
                          concat=torch.cat((_unit(h_n), _unit(h_c), _unit(h_s)), -1))
        available = dict(nucleus=nucleus_available, cytoplasm=cytoplasm_available, local=local_available,
                         spatial=spatial_available, fused=local_available | spatial_available)
        edge_count = graph.sum()
        diag = {f"{name}_available_fraction": mask.float().mean().detach() for name, mask in available.items()}
        diag.update(paired_fraction=paired.float().mean().detach(), physical_graph_edges=edge_count.detach(),
                    spatial_attention_entropy=(-(read_attention * read_attention.clamp_min(1e-8).log()).sum(-1)).mean().detach(),
                    lr_strength=self.spatial_encoder.lr_strength.detach(),
                    fusion_residual_strength=(gate.squeeze(-1) * paired).sum().detach() / paired.sum().clamp_min(1),
                    local_nucleus_weight=(local_weight.squeeze(-1) * local_available).sum().detach() / local_available.sum().clamp_min(1))
        for key, mask in (("neighbor_lr_coverage", graph), ("center_lr_coverage", present)):
            coverage = batch.get(key)
            if coverage is not None:
                if coverage.shape != mask.shape or not bool(torch.isfinite(coverage[mask]).all()):
                    raise ValueError(f"{key} must align with finite eligible LR evidence")
                diag["lr_coverage_on_edges" if key == "neighbor_lr_coverage" else "center_lr_coverage"] = (
                    torch.where(mask, coverage, 0.).sum() / mask.sum().clamp_min(1)).detach()
        result = dict(embeddings=embeddings, mode_available=available, diagnostics=diag,
                      graph_attention=graph_attention, readout_attention=read_attention,
                      conditioned_readout_attention=condition_attention, neighbor_graph=graph)
        if return_embeddings:
            return result
        query_genes, query_valid = batch["query_gene_ids"], batch["query_valid_mask"].bool()
        if query_genes.ndim != 2 or query_genes.shape[0] != size or query_valid.shape != query_genes.shape:
            raise ValueError("Queries require aligned [B,Q] IDs and validity")
        if bool((query_valid & (query_genes.eq(self.pad_id) | query_genes.eq(self.cls_id))).any()):
            raise ValueError("Valid queries cannot be PAD or CLS")
        query = _unit(self.backbone.encoder(torch.where(query_valid, query_genes, self.pad_id)))
        # Count targets are deliberately never inspected by the model.
        result["count_parameters"] = {
            name: self.joint_decoder(query, embeddings[name][:, None].expand(-1, query.shape[1], -1))
            for name in MODES}
        return result


def build_model(cfg: Mapping, backbone: nn.Module | None = None) -> SubcellularV12:
    return SubcellularV12(cfg.get("assets_dir", cfg.get("asset_dir", ".")), backbone=backbone,
                         **dict(cfg.get("model", {})))
