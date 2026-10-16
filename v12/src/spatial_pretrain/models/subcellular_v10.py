"""V10: independent nuclear and neighbour views predict a fixed whole-cell CLS.

The target is ordinary pretrained scGPT applied to unmasked N+C counts. Its
encoder is permanently frozen and remains in eval mode. The context predictor
uses only neighbouring cytoplasm and relative coordinates: its query is a
learned parameter, never a center-cell feature. No center expression or teacher
feature can enter the context prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .scgpt_backbone import ScGPTBackbone


def _unit(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-8)


class ResidualPredictor(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, expansion * dim),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(expansion * dim, dim))
        # The nuclear branch starts from the unchanged frozen nuclear CLS.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, value: Tensor) -> Tensor:
        return value + self.network(value)


class NeighborPredictor(nn.Module):
    """One learned query attends to a set of neighbour cytoplasmic CLSs."""
    def __init__(self, dim: int, heads: int, coordinate_scale: float, dropout: float):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError("spatial_heads must divide the backbone dimension")
        if coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.coordinate_scale = float(coordinate_scale)
        self.query = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.normal_(self.query, std=self.head_dim ** -.5)
        self.norm = nn.LayerNorm(dim)
        self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.position_bias = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads))
        self.output = nn.Linear(dim, dim)
        self.predictor = ResidualPredictor(dim, dropout=dropout)

    def forward(self, neighbors: Tensor, present: Tensor, relative_coords: Tensor):
        size, count, _ = neighbors.shape
        if not count:
            # Keep a differentiable zero in the graph when a batch has no neighbours.
            zero = sum(parameter.reshape(-1)[0] * 0. for parameter in self.parameters())
            zero = zero + neighbors.new_zeros((size, self.dim))
            return zero, neighbors.new_zeros((size, self.heads, 0))
        features = self.norm(neighbors)
        keys = self.key(features).reshape(size, count, self.heads, self.head_dim)
        values = self.value(features).reshape(size, count, self.heads, self.head_dim)
        # Invalid coordinates must not introduce NaNs even when their weight is zero.
        coords = torch.where(present[..., None], relative_coords.float(), 0.)
        coords = coords / self.coordinate_scale
        coords = coords.sign() * torch.log1p(coords.abs())
        bias = self.position_bias(coords).permute(0, 2, 1)
        logits = torch.einsum("hd,bkhd->bhk", self.query, keys) * self.head_dim ** -.5 + bias
        valid = present[:, None, :]
        # finfo.min (rather than -inf) gives finite softmax for an all-missing row.
        weights = logits.float().masked_fill(~valid, torch.finfo(torch.float32).min).softmax(-1)
        weights = weights * valid
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        message = torch.einsum("bhk,bkhd->bhd", weights.to(values.dtype), values).reshape(size, self.dim)
        prediction = self.predictor(self.output(message))
        prediction = prediction * present.any(-1, keepdim=True)
        return prediction, weights


class SubcellularV10(nn.Module):
    def __init__(self, asset_dir: str | Path, neighbors: int = 16, checkpointing: bool = False,
                 encoder_chunk_size: int = 64, *, backbone_kind: str = "scgpt",
                 backbone: nn.Module | None = None, spatial_heads: int | None = None,
                 coordinate_scale: float = 50., predictor_dropout: float = 0.):
        super().__init__()
        if backbone_kind != "scgpt":
            raise ValueError("V10 requires ordinary scGPT, never scGPT-spatial")
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 0:
            raise ValueError("neighbors must be a nonnegative integer")
        if isinstance(encoder_chunk_size, bool) or not isinstance(encoder_chunk_size, int) or encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be a positive integer")
        if not 0 <= predictor_dropout < 1:
            raise ValueError("predictor_dropout must be in [0, 1)")
        self.backbone = backbone if backbone is not None else ScGPTBackbone(
            asset_dir, checkpointing=False, backbone_kind="scgpt")
        if getattr(self.backbone, "backbone_kind", "scgpt") != "scgpt":
            raise ValueError("Injected V10 backbone must identify as ordinary scGPT")
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.vocab, self.dim = self.backbone.vocab, self.backbone.dim
        self.pad_id, self.cls_id = self.backbone.pad_id, self.vocab["<cls>"]
        self.pad_value = float(getattr(self.backbone, "config", {}).get("pad_value", -2.))
        self.neighbors, self.encoder_chunk_size = neighbors, encoder_chunk_size
        self.nucleus_predictor = ResidualPredictor(self.dim, dropout=predictor_dropout)
        heads = self.backbone.nheads if spatial_heads is None else spatial_heads
        self.context_predictor = NeighborPredictor(self.dim, heads, coordinate_scale, predictor_dropout)
        self.arch_config = dict(
            model_type="subcellular_fixed_teacher_two_view_v10", backbone_kind="scgpt",
            backbone_dim=self.dim, backbone_trainable_layers=0, teacher="fixed_unmasked_whole_cell_cls",
            nucleus_view="frozen_nuclear_cls_plus_residual_predictor",
            context_view="learned_query_neighbour_cytoplasm_only", spatial_heads=heads,
            neighbors=neighbors, coordinate_scale=float(coordinate_scale),
            predictor_dropout=float(predictor_dropout), position="relative_xy_log1p_attention_bias",
            readouts=["base", "nucleus", "context", "joint", "base_augmented"],
            base_readout="l2_frozen_cls", predicted_readouts="l2_normalized",
            base_augmented="concat(unit_base,unit_nucleus,unit_context)",
            feature_cache="raw_frozen_cls_before_predictors_and_l2",
            missing_nucleus="excluded_from_nuclear_and_paired_losses;zero_readout",
            checkpointing=False)
        # Accepted for launch compatibility; no gradient checkpointing is needed
        # for a permanently frozen encoder.
        del checkpointing

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @property
    def backbone_trainable_layers(self) -> int:
        return 0

    @torch.no_grad()
    def encode_sequences(self, genes: Tensor, bins: Tensor, valid: Tensor) -> Tensor:
        if genes.ndim != 2 or genes.shape != bins.shape or genes.shape != valid.shape or genes.shape[1] < 1:
            raise ValueError("Each token sequence needs [N, L] genes/bins/valid with L >= 1")
        if genes.shape[0] == 0:
            return bins.new_empty((0, self.dim), dtype=torch.float32)
        if not bool((genes[:, 0].eq(self.cls_id) & valid[:, 0].bool()).all()):
            raise ValueError("Every sequence must start with a valid CLS token")
        visible = valid.bool() & genes.ne(self.pad_id)
        genes = torch.where(visible, genes, self.pad_id)
        bins = torch.where(visible, bins, self.pad_value).float()
        result = []
        for start in range(0, len(genes), self.encoder_chunk_size):
            part = slice(start, start + self.encoder_chunk_size)
            addition = bins.new_zeros((*genes[part].shape, self.dim))
            hidden = self.backbone(genes[part], bins[part], ~visible[part], addition)
            result.append(hidden[:, 0].float())
        return torch.cat(result)

    @torch.no_grad()
    def encode_views(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Encode token views once; return reusable, unnormalized frozen CLSs.

        The returned ``teacher_features`` and ``nucleus_features`` are [B,D];
        ``neighbor_features`` is [B,K,D], with zero at absent neighbors. Cache
        consumers must preserve the matching ``neighbor_present``,
        ``relative_coords`` and ``nucleus_present`` metadata. They can pass
        these raw features to forward without any token fields. Cache identity
        and provenance checks belong to the caller, not the predictor.
        """
        genes, bins = batch["neighbor_gene_ids"], batch["neighbor_bins"]
        valid, present = batch["neighbor_valid"], batch["neighbor_present"].bool()
        if genes.ndim != 3 or genes.shape != bins.shape or genes.shape != valid.shape:
            raise ValueError("Neighbor tokens must have shape [B, K, L]")
        size, count, _ = genes.shape
        if present.shape != (size, count):
            raise ValueError("Neighbor presence must match [B, K]")
        nucleus = self.encode_sequences(batch["nucleus_gene_ids"], batch["nucleus_bins"], batch["nucleus_valid"])
        base = self.encode_sequences(batch["teacher_gene_ids"], batch["teacher_bins"], batch["teacher_valid"])
        if len(nucleus) != size or len(base) != size:
            raise ValueError("All V10 views must have the same center-cell batch size")
        # Skip absent neighbours altogether. Their token arrays may be arbitrary
        # padding and cannot influence either features or normalization.
        neighbor_features = base.new_zeros((size, count, self.dim))
        if bool(present.any()):
            neighbor_features[present] = self.encode_sequences(genes[present], bins[present], valid[present])
        return dict(teacher_features=base, nucleus_features=nucleus,
                    neighbor_features=neighbor_features)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False):
        del return_embeddings  # Same no-decoder result for training and evaluation.
        feature_keys = {"teacher_features", "nucleus_features", "neighbor_features"}
        supplied = feature_keys.intersection(batch)
        if supplied and supplied != feature_keys:
            raise ValueError("Supply all three raw CLS feature fields or none")
        features = batch if supplied else self.encode_views(batch)
        base, nucleus, neighbor_features = (features[key] for key in (
            "teacher_features", "nucleus_features", "neighbor_features"))
        present, coords = batch["neighbor_present"].bool(), batch["relative_coords"]
        if present.ndim != 2:
            raise ValueError("Neighbor presence must have shape [B, K]")
        size, count = present.shape
        if (base.shape != (size, self.dim) or nucleus.shape != (size, self.dim)
                or neighbor_features.shape != (size, count, self.dim)):
            raise ValueError("Raw CLS features must match [B,D], [B,D], [B,K,D]")
        if coords.shape != (size, count, 2):
            raise ValueError("Neighbor coordinates must match [B, K, 2]")
        for name in feature_keys:
            value = features[name]
            if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must contain finite floating-point raw CLS features")
            if value.device != present.device:
                raise ValueError("Raw CLS features and neighbor presence must share a device")
        if coords.device != present.device or not bool(torch.isfinite(coords[present]).all()):
            raise ValueError("Present neighbors need finite coordinates on the features device")
        nucleus_available = batch.get("nucleus_present", torch.ones(
            size, dtype=torch.bool, device=present.device)).bool()
        if nucleus_available.shape != (size,) or nucleus_available.device != present.device:
            raise ValueError("nucleus_present must have shape [B] on the features device")
        # Cached features are fixed targets/inputs just like encode_views output;
        # they must never introduce a gradient path into the teacher encoder.
        base, nucleus, neighbor_features = (value.detach().float()
                                             for value in (base, nucleus, neighbor_features))
        context_raw, attention = self.context_predictor(neighbor_features, present, coords)
        nucleus_prediction = _unit(self.nucleus_predictor(nucleus)) * nucleus_available[:, None]
        available = present.any(-1)
        context_prediction = _unit(context_raw) * available[:, None]
        teacher = _unit(base).detach()
        joint = _unit(nucleus_prediction + context_prediction)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
        diagnostics = dict(
            nucleus_available_fraction=nucleus_available.float().mean().detach(),
            context_available_fraction=available.float().mean().detach(),
            context_attention_entropy=entropy.mean().detach(),
            nucleus_target_cosine=((nucleus_prediction * teacher).sum(-1) * nucleus_available).sum().detach()
            / nucleus_available.sum().clamp_min(1),
            context_target_cosine=((context_prediction * teacher).sum(-1) * available).sum().detach()
            / available.sum().clamp_min(1))
        return dict(
            embeddings=dict(base=teacher, nucleus=nucleus_prediction, context=context_prediction,
                            joint=joint, base_augmented=torch.cat((teacher, nucleus_prediction,
                                                                  context_prediction), dim=-1)),
            teacher=teacher, nucleus_prediction=nucleus_prediction, context_prediction=context_prediction,
            nucleus_available=nucleus_available, context_available=available, diagnostics=diagnostics)


def build_model(cfg: Mapping, backbone: nn.Module | None = None) -> SubcellularV10:
    model = dict(cfg.get("model", {}))
    return SubcellularV10(cfg.get("assets_dir", cfg.get("asset_dir", ".")), backbone=backbone, **model)
