"""V11: independent nuclear and LR-informed neighbourhood views, learned fusion.

The shared ordinary-scGPT backbone encodes nuclear and neighbouring whole-cell
RNA separately. Only the learned fusion sees both views. A single gene-ID
conditioned decoder reconstructs center-cell expression bins from each of the
three readouts. There is no whole-cell teacher, latent alignment or EMA network.

Neighbour LR matrices have receiver-by-sender orientation. A physical-radius
AND KNN graph masks neighbour-to-neighbour attention before applying the LR
soft bias; a learned center query subsequently reads those neighbour features.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .scgpt_backbone import ScGPTBackbone
from .subcellular_v4 import JointBinDecoder


MODES = ("nucleus", "spatial", "fused")
READOUTS = (*MODES, "concat")


def _unit(value: Tensor) -> Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-8)


def _masked_softmax(logits: Tensor, valid: Tensor) -> Tensor:
    weights = logits.float().masked_fill(~valid, torch.finfo(torch.float32).min).softmax(-1)
    weights = weights * valid
    return weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)


def _positive(name: str, value: float):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


class LRNeighborEncoder(nn.Module):
    """Directed physical graph attention followed by a center-relative readout."""

    def __init__(self, dim: int, heads: int, *, coordinate_scale: float = 50.,
                 spatial_radius: float = 100., graph_neighbors: int = 8,
                 lr_enabled: bool = True, lr_max_strength: float = 2.,
                 lr_initial_strength: float = .1, dropout: float = 0.):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError("spatial_heads must divide the backbone dimension")
        _positive("coordinate_scale", coordinate_scale)
        _positive("spatial_radius", spatial_radius)
        if isinstance(graph_neighbors, bool) or not isinstance(graph_neighbors, int) or graph_neighbors < 1:
            raise ValueError("graph_neighbors must be a positive integer")
        if not isinstance(lr_enabled, bool):
            raise ValueError("lr_enabled must be boolean")
        if (not math.isfinite(lr_max_strength) or not math.isfinite(lr_initial_strength)
                or not 0 < lr_initial_strength < lr_max_strength):
            raise ValueError("LR strength must satisfy finite 0 < initial < max")
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.coordinate_scale, self.spatial_radius = float(coordinate_scale), float(spatial_radius)
        self.graph_neighbors, self.lr_enabled = graph_neighbors, lr_enabled
        self.lr_max_strength = float(lr_max_strength)
        ratio = lr_initial_strength / lr_max_strength
        self.lr_scale_logit = nn.Parameter(torch.full((heads,), math.log(ratio / (1 - ratio))),
                                          requires_grad=lr_enabled)
        self.norm = nn.LayerNorm(dim)
        self.query, self.key, self.value = (nn.Linear(dim, dim) for _ in range(3))
        self.edge_position = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads))
        self.output = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(2 * dim, dim))
        self.readout_query = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.normal_(self.readout_query, std=self.head_dim ** -.5)
        self.readout_norm = nn.LayerNorm(dim)
        self.readout_key, self.readout_value = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.center_position = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, heads))
        self.readout_output = nn.Linear(dim, dim)

    @property
    def lr_strength(self) -> Tensor:
        return self.lr_max_strength * self.lr_scale_logit.sigmoid()

    def _coordinates(self, coords: Tensor) -> Tensor:
        coords = coords / self.coordinate_scale
        return coords.sign() * torch.log1p(coords.abs())

    def graph(self, present: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Return receiver-by-sender eligibility and sender-minus-receiver XY."""
        size, count = present.shape
        clean = torch.where(present[..., None], coords.float(), 0.)
        relative = clean[:, None, :, :] - clean[:, :, None, :]
        distance = relative.square().sum(-1).sqrt()
        eligible = present[:, :, None] & present[:, None, :]
        eligible = eligible & ~torch.eye(count, device=present.device, dtype=torch.bool)[None]
        eligible = eligible & distance.le(self.spatial_radius)
        if count > self.graph_neighbors:
            ranked = distance.masked_fill(~eligible, torch.inf)
            indices = ranked.topk(self.graph_neighbors, dim=-1, largest=False).indices
            knn = torch.zeros_like(eligible).scatter_(-1, indices, True)
            eligible = eligible & knn
        return eligible, relative

    def forward(self, neighbors: Tensor, present: Tensor, relative_coords: Tensor,
                lr_bias: Tensor | None = None):
        size, count, _ = neighbors.shape
        if not count:
            zero = sum(p.reshape(-1)[0] * 0. for p in self.parameters() if p.requires_grad)
            return (neighbors.new_zeros((size, self.dim)) + zero,
                    neighbors.new_zeros((size, self.heads, 0, 0)),
                    neighbors.new_zeros((size, self.heads, 0)),
                    torch.zeros((size, 0, 0), dtype=torch.bool, device=neighbors.device))
        eligible, offsets = self.graph(present, relative_coords)
        features = self.norm(neighbors)
        q, k, v = [layer(features).reshape(size, count, self.heads, self.head_dim)
                   for layer in (self.query, self.key, self.value)]
        # Keep small LR perturbations and softmax in float32 under bf16 AMP.
        with torch.autocast(device_type=neighbors.device.type, enabled=False):
            logits = torch.einsum("bihd,bjhd->bhij", q.float(), k.float()) * self.head_dim ** -.5
        logits = logits + self.edge_position(self._coordinates(offsets)).float().permute(0, 3, 1, 2)
        if self.lr_enabled:
            if lr_bias is None or lr_bias.shape != (size, count, count):
                raise ValueError("LR-enabled V11 needs neighbor_lr_bias [B,K,K], receiver by sender")
            if not bool(torch.isfinite(lr_bias[eligible]).all()):
                raise ValueError("Eligible LR edges must contain finite evidence")
            evidence = torch.where(eligible, lr_bias.float(), 0.).clamp(0., 1.)
            logits = logits + evidence[:, None] * self.lr_strength[None, :, None, None].float()
        attention = _masked_softmax(logits, eligible[:, None])
        messages = torch.einsum("bhij,bjhd->bihd", attention.to(v.dtype), v).reshape(size, count, self.dim)
        # No output-bias update for isolated cells; their original cell features remain.
        states = neighbors + self.output(messages) * eligible.any(-1, keepdim=True)
        states = (states + self.ffn(states)) * present[..., None]
        normalized = self.readout_norm(states)
        keys, values = [layer(normalized).reshape(size, count, self.heads, self.head_dim)
                        for layer in (self.readout_key, self.readout_value)]
        clean_coords = torch.where(present[..., None], relative_coords.float(), 0.)
        with torch.autocast(device_type=neighbors.device.type, enabled=False):
            read_logits = torch.einsum("hd,bkhd->bhk", self.readout_query.float(), keys.float())
        read_logits = read_logits * self.head_dim ** -.5
        read_logits = read_logits + self.center_position(self._coordinates(clean_coords)).float().permute(0, 2, 1)
        read_attention = _masked_softmax(read_logits, present[:, None])
        pooled = torch.einsum("bhk,bkhd->bhd", read_attention.to(values.dtype), values).reshape(size, self.dim)
        spatial = self.readout_output(pooled) * present.any(-1, keepdim=True)
        return spatial.float(), attention, read_attention, eligible


class SubcellularV11(nn.Module):
    num_bins = 51

    def __init__(self, asset_dir: str | Path, neighbors: int = 8, checkpointing: bool = True,
                 encoder_chunk_size: int = 32, *, backbone_kind: str = "scgpt",
                 backbone: nn.Module | None = None, backbone_trainable_layers: int | None = None,
                 spatial_heads: int | None = None, coordinate_scale: float = 50.,
                 spatial_radius: float = 100., graph_neighbors: int = 8,
                 max_neighbor_distance: float | None = None, lr_enabled: bool = True,
                 lr_max_strength: float = 2., lr_initial_strength: float = .1,
                 dropout: float = 0.):
        super().__init__()
        if backbone_kind != "scgpt":
            raise ValueError("V11 requires ordinary scGPT, never scGPT-spatial")
        if isinstance(neighbors, bool) or not isinstance(neighbors, int) or neighbors < 0:
            raise ValueError("neighbors must be a nonnegative integer")
        if isinstance(encoder_chunk_size, bool) or not isinstance(encoder_chunk_size, int) or encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size must be a positive integer")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if max_neighbor_distance is not None:
            _positive("max_neighbor_distance", max_neighbor_distance)
        self.backbone = backbone if backbone is not None else ScGPTBackbone(
            asset_dir, checkpointing=False, backbone_kind="scgpt")
        if getattr(self.backbone, "backbone_kind", "scgpt") != "scgpt":
            raise ValueError("Injected V11 backbone must be ordinary scGPT")
        # Chunk-level checkpointing avoids retaining every cell's encoder graph;
        # do not nest per-layer checkpointing inside it.
        self.backbone.checkpointing = False
        self.vocab, self.dim = self.backbone.vocab, self.backbone.dim
        self.pad_id, self.cls_id = self.backbone.pad_id, self.vocab["<cls>"]
        self.pad_value = float(getattr(self.backbone, "config", {}).get("pad_value", -2.))
        self.neighbors, self.encoder_chunk_size = neighbors, encoder_chunk_size
        self.checkpointing, self.max_neighbor_distance = bool(checkpointing), max_neighbor_distance
        heads = self.backbone.nheads if spatial_heads is None else spatial_heads
        self.spatial_encoder = LRNeighborEncoder(
            self.dim, heads, coordinate_scale=coordinate_scale, spatial_radius=spatial_radius,
            graph_neighbors=graph_neighbors, lr_enabled=lr_enabled,
            lr_max_strength=lr_max_strength, lr_initial_strength=lr_initial_strength, dropout=dropout)
        self.fusion_nucleus_norm, self.fusion_spatial_norm = nn.LayerNorm(self.dim), nn.LayerNorm(self.dim)
        self.fusion = nn.Sequential(nn.Linear(2 * self.dim, 2 * self.dim), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * self.dim, self.dim))
        self.joint_decoder = JointBinDecoder(self.dim, self.dim, self.num_bins)
        self.arch_config = dict(
            model_type="subcellular_complementary_fusion_v11", backbone_kind="scgpt",
            backbone_dim=self.dim, neighbors=neighbors, spatial_heads=heads,
            coordinate_scale=float(coordinate_scale), spatial_radius=float(spatial_radius),
            graph_neighbors=graph_neighbors, max_neighbor_distance=max_neighbor_distance,
            lr_enabled=lr_enabled, lr_max_strength=float(lr_max_strength),
            lr_initial_strength=float(lr_initial_strength), lr_orientation="receiver_sender",
            lr_coverage="diagnostic_only", decoder="one_shared_gene_conditioned_51_bin_head",
            num_bins=self.num_bins, nucleus_view="nucleus_only_cls",
            spatial_view="neighbors_whole_RNA_physical_graph_then_learned_query_readout",
            fusion="MLP(concat(LN_hN,LN_hS));missing_view_falls_back_to_available_view",
            readouts=list(READOUTS), concat="concat(unit_hN,unit_hS)",
            checkpointing=self.checkpointing, encoder_chunk_size=encoder_chunk_size,
            dropout=float(dropout), feature_cache=False)
        total_layers = len(self.backbone.transformer_encoder.layers)
        self.set_backbone_trainable_layers(total_layers if backbone_trainable_layers is None
                                          else backbone_trainable_layers)

    @property
    def backbone_trainable_layers(self) -> int:
        return self._backbone_trainable_layers

    def set_backbone_trainable_layers(self, count: int):
        """Train the last count Transformer layers; preserve pretrained token embeddings."""
        layers = self.backbone.transformer_encoder.layers
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(layers):
            raise ValueError(f"count must be an integer between 0 and {len(layers)}")
        self.backbone.requires_grad_(False)
        if count:
            for layer in layers[-count:]:
                layer.requires_grad_(True)
        self._backbone_trainable_layers = count
        self.arch_config["backbone_trainable_layers"] = count
        self._apply_backbone_modes(self.training)

    def _apply_backbone_modes(self, mode: bool):
        self.backbone.eval()
        if mode and self._backbone_trainable_layers:
            for layer in self.backbone.transformer_encoder.layers[-self._backbone_trainable_layers:]:
                layer.train(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_backbone_modes(mode)
        return self

    def _cls(self, genes: Tensor, bins: Tensor, visible: Tensor) -> Tensor:
        addition = bins.new_zeros((*genes.shape, self.dim))
        return self.backbone(genes, bins, ~visible, addition)[:, 0].float()

    def encode_sequences(self, genes: Tensor, bins: Tensor, valid: Tensor) -> Tensor:
        if genes.ndim != 2 or genes.shape != bins.shape or genes.shape != valid.shape or genes.shape[1] < 1:
            raise ValueError("Tokens require [N,L] genes, bins, validity with L >= 1")
        if not len(genes):
            return bins.new_empty((0, self.dim), dtype=torch.float32)
        if not bool((genes[:, 0].eq(self.cls_id) & valid[:, 0].bool()).all()):
            raise ValueError("Every present sequence must start with a valid CLS token")
        visible = valid.bool() & genes.ne(self.pad_id)
        if not bool(torch.isfinite(bins[visible]).all()):
            raise ValueError("Visible input bins must be finite")
        clean_genes = torch.where(visible, genes, self.pad_id)
        clean_bins = torch.where(visible, bins, self.pad_value).float()
        outputs = []
        for start in range(0, len(genes), self.encoder_chunk_size):
            part = slice(start, start + self.encoder_chunk_size)
            args = (clean_genes[part], clean_bins[part], visible[part])
            if self.checkpointing and self.training and torch.is_grad_enabled() and self.backbone_trainable_layers:
                outputs.append(checkpoint(self._cls, *args, use_reentrant=False))
            else:
                outputs.append(self._cls(*args))
        return torch.cat(outputs)

    def _neighbor_presence(self, batch: Mapping[str, Tensor], coords: Tensor) -> Tensor:
        present = batch["neighbor_present"].bool().clone()
        if coords.shape != (*present.shape, 2):
            raise ValueError("Relative coordinates must have shape [B,K,2]")
        if not bool(torch.isfinite(coords[present]).all()):
            raise ValueError("Present neighbours require finite physical coordinates")
        ids = batch.get("neighbor_cell_identity")
        if ids is not None:
            centers = batch.get("cell_identity")
            if ids.shape != (*present.shape, 2) or centers is None or centers.shape != (len(present), 2):
                raise ValueError("Neighbor/center identities require [B,K,2] and [B,2]")
            present &= ~ids.eq(centers[:, None]).all(-1)
            same = ids[:, :, None, :].eq(ids[:, None, :, :]).all(-1)
            # Keep first present instance only; absent duplicate slots do not suppress real cells.
            earlier = torch.ones_like(same).tril(diagonal=-1)
            duplicate = (same & earlier & present[:, None, :]).any(-1)
            present &= ~duplicate
        if self.max_neighbor_distance is not None:
            present &= coords.float().square().sum(-1).le(self.max_neighbor_distance ** 2)
        return present

    def _input_presence(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        present = self._neighbor_presence(batch, batch["relative_coords"])
        genes, valid = batch["neighbor_gene_ids"], batch["neighbor_valid"]
        if genes.ndim != 3 or valid.shape != genes.shape or genes.shape[:2] != present.shape:
            raise ValueError("Neighbor token validity must match [B,K,L] and presence [B,K]")
        nucleus_genes, nucleus_valid = batch["nucleus_gene_ids"], batch["nucleus_valid"]
        if nucleus_genes.ndim != 2 or nucleus_genes.shape[0] != len(present) or nucleus_valid.shape != nucleus_genes.shape:
            raise ValueError("Nuclear gene IDs and validity must align to [B,L]")
        nucleus_available = batch.get("nucleus_present", nucleus_valid[:, 1:].any(-1)).bool()
        if nucleus_available.shape != (len(present),):
            raise ValueError("nucleus_present must be [B]")
        nucleus_available = nucleus_available & (nucleus_valid.bool() & nucleus_genes.ne(self.pad_id)
                                                  & nucleus_genes.ne(self.cls_id)).any(-1)
        present = present & (valid.bool() & genes.ne(self.pad_id) & genes.ne(self.cls_id)).any(-1)
        return nucleus_available, present

    def mode_availability(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Exact input-only masks for whole-update DDP loss counts before forward."""
        nucleus, present = self._input_presence(batch)
        spatial = present.any(-1)
        return dict(nucleus=nucleus, spatial=spatial, fused=nucleus | spatial)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False):
        if any(key in batch for key in ("teacher_features", "nucleus_features", "neighbor_features")):
            raise ValueError("V11 trains the encoder from tokens; frozen feature caches are unsupported")
        genes, bins, valid = (batch[key] for key in ("neighbor_gene_ids", "neighbor_bins", "neighbor_valid"))
        if genes.ndim != 3 or genes.shape != bins.shape or genes.shape != valid.shape:
            raise ValueError("Neighbor tokens must share shape [B,K,L]")
        size, count, _ = genes.shape
        coords = batch["relative_coords"]
        if batch["neighbor_present"].shape != (size, count):
            raise ValueError("Neighbor presence must match token shape [B,K]")
        nucleus_available, present = self._input_presence(batch)
        nucleus_genes, nucleus_bins, nucleus_valid = [batch[f"nucleus_{name}"] for name in ("gene_ids", "bins", "valid")]
        if nucleus_genes.ndim != 2 or nucleus_genes.shape[0] != size:
            raise ValueError("Nuclear tokens must align to [B,L]")
        h_n = bins.new_zeros((size, self.dim), dtype=torch.float32)
        h_neighbors = bins.new_zeros((size, count, self.dim), dtype=torch.float32)
        if bool(nucleus_available.any()):
            h_n[nucleus_available] = self.encode_sequences(nucleus_genes[nucleus_available],
                nucleus_bins[nucleus_available], nucleus_valid[nucleus_available])
        if bool(present.any()):
            h_neighbors[present] = self.encode_sequences(genes[present], bins[present], valid[present])
        if not bool(nucleus_available.any() | present.any()):
            # Allow a zero-loss backward even on ranks with no usable input.
            h_n = h_n + sum(p.reshape(-1)[0] * 0. for p in self.backbone.parameters() if p.requires_grad)
        h_s, graph_attention, read_attention, graph = self.spatial_encoder(
            h_neighbors, present, coords, batch.get("neighbor_lr_bias"))
        spatial_available = present.any(-1)
        h_f = self.fusion(torch.cat((self.fusion_nucleus_norm(h_n), self.fusion_spatial_norm(h_s)), -1))
        paired = nucleus_available & spatial_available
        h_f = torch.where(paired[:, None], h_f,
                          torch.where(nucleus_available[:, None], h_n, h_s))
        available = dict(nucleus=nucleus_available, spatial=spatial_available,
                         fused=nucleus_available | spatial_available)
        embeddings = dict(nucleus=h_n, spatial=h_s, fused=h_f,
                          concat=torch.cat((_unit(h_n), _unit(h_s)), -1))
        edge_count = graph.sum()
        diag = dict(nucleus_available_fraction=nucleus_available.float().mean().detach(),
                    spatial_available_fraction=spatial_available.float().mean().detach(),
                    paired_fraction=paired.float().mean().detach(),
                    physical_graph_edges=edge_count.detach(),
                    spatial_attention_entropy=(-(read_attention * read_attention.clamp_min(1e-8).log())
                                               .sum(-1)).mean().detach(),
                    lr_strength=self.spatial_encoder.lr_strength.detach())
        coverage = batch.get("neighbor_lr_coverage")
        if coverage is not None:
            if coverage.shape != graph.shape or not bool(torch.isfinite(coverage[graph]).all()):
                raise ValueError("LR coverage must be finite on graph edges and have shape [B,K,K]")
            diag["lr_coverage_on_edges"] = (torch.where(graph, coverage, 0.).sum()
                                             / edge_count.clamp_min(1)).detach()
        result = dict(embeddings=embeddings, mode_available=available, diagnostics=diag,
                      graph_attention=graph_attention, readout_attention=read_attention,
                      neighbor_graph=graph)
        if return_embeddings:
            return result
        query_genes = batch["query_gene_ids"]
        query_valid = batch["query_valid_mask"].bool()
        if query_genes.ndim != 2 or query_genes.shape[0] != size or query_valid.shape != query_genes.shape:
            raise ValueError("Gene queries need aligned [B,Q] gene IDs and validity")
        if bool((query_valid & (query_genes.eq(self.pad_id) | query_genes.eq(self.cls_id))).any()):
            raise ValueError("Valid gene queries cannot be PAD or CLS")
        safe_queries = torch.where(query_valid, query_genes, self.pad_id)
        # Only gene identity enters D, never target bins/counts.
        query = _unit(self.backbone.encoder(safe_queries))
        result["logits"] = {mode: self.joint_decoder(query, embeddings[mode][:, None].expand(-1, query.shape[1], -1))
                            for mode in MODES}
        return result


def build_model(cfg: Mapping, backbone: nn.Module | None = None) -> SubcellularV11:
    return SubcellularV11(cfg.get("assets_dir", cfg.get("asset_dir", ".")), backbone=backbone,
                         **dict(cfg.get("model", {})))
