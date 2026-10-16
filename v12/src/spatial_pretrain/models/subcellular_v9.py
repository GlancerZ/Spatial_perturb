"""V9: one-hop, two-view reconstruction of the center cell.

Every cell (center + K neighbours) is one ordinary-scGPT sequence
``[CLS, nuclear tokens, cytoplasmic tokens]``; tokens add a zero-initialized
compartment embedding (0 nucleus, 1 cytoplasm, 2 CLS/PAD). The CLS output is
the cell vector.

* self view:    z_self = CLS(center visible tokens)
* context view: z_ctx  = one-hop attention with query z_self over the K
  neighbour CLS vectors, logits = QK + distance bias + bounded LR bias.
  Values come only from neighbours, so z_ctx carries neighbour evidence.

One shared 51-bin decoder reads (query gene, z) for each view. Downstream
readouts: ``self``, ``ctx`` and ``joint`` = [z_self; z_ctx].
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .scgpt_backbone import ScGPTBackbone
from .subcellular_v4 import JointBinDecoder
from .subcellular_v6 import LRSpatialAttention


class SubcellularV9(nn.Module):
    num_bins = 51

    def __init__(self, asset_dir: str | Path, neighbors: int = 16, checkpointing: bool = True,
                 encoder_chunk_size: int = 64, *, backbone_kind: str = "scgpt",
                 backbone: nn.Module | None = None, lr_enabled: bool = True,
                 lr_max_strength: float = 2., lr_initial_strength: float = .1):
        super().__init__()
        if backbone_kind != "scgpt":
            raise ValueError("V9 uses the ordinary scGPT backbone")
        for name, value in dict(neighbors=neighbors, encoder_chunk_size=encoder_chunk_size).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.backbone = backbone if backbone is not None else ScGPTBackbone(
            asset_dir, checkpointing=checkpointing, backbone_kind="scgpt")
        self.vocab = self.backbone.vocab
        self.pad_id, self.cls_id = self.backbone.pad_id, self.vocab["<cls>"]
        self.dim = self.backbone.dim
        self.neighbors, self.encoder_chunk_size = neighbors, encoder_chunk_size
        self.checkpointing = bool(checkpointing)
        self.compartment_embedding = nn.Embedding(3, self.dim)
        nn.init.zeros_(self.compartment_embedding.weight)
        heads = self.backbone.nheads
        self.spatial = LRSpatialAttention(self.dim, heads, lr_enabled=lr_enabled,
                                          lr_max_strength=lr_max_strength,
                                          lr_initial_strength=lr_initial_strength)
        self.joint_decoder = JointBinDecoder(self.dim, self.dim, self.num_bins)
        self.lr_enabled = lr_enabled
        self._backbone_trainable_layers = 0
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.arch_config = dict(
            model_type="subcellular_two_view_one_hop_v9", backbone_kind=backbone_kind,
            backbone_dim=self.dim, neighbors=neighbors, spatial_heads=heads,
            cell_vector="scgpt_cls_of_joint_nuclear_cytoplasm_sequence",
            compartment_embedding="zero_init_nucleus_cytoplasm_cls",
            context_view="one_hop_attention_query_self_values_neighbours_only",
            bias="distance_mlp_plus_bounded_lr", lr_enabled=lr_enabled,
            lr_max_strength=float(lr_max_strength), lr_initial_strength=float(lr_initial_strength),
            decoder="shared_51_bin_decoder_for_both_views", num_bins=self.num_bins,
            readouts=["self", "ctx", "joint"], backbone_trainable_layers=0)

    # ---- backbone fine-tuning (same contract as V7) ----
    @property
    def backbone_trainable_layers(self) -> int:
        return self._backbone_trainable_layers

    def _apply_backbone_modes(self, mode: bool):
        count = self._backbone_trainable_layers
        if not count or not mode:
            self.backbone.eval()
            return
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
        if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= len(layers):
            raise ValueError(f"count must be an integer between 0 and {len(layers)}")
        self.backbone.requires_grad_(False)
        if count:
            for layer in layers[-count:]:
                layer.requires_grad_(True)
        self._backbone_trainable_layers = count
        self.arch_config["backbone_trainable_layers"] = count
        self._apply_backbone_modes(self.training)

    # ---- encoding ----
    def _cls(self, genes: Tensor, bins: Tensor, valid: Tensor, compartments: Tensor) -> Tensor:
        addition = self.compartment_embedding(compartments.clamp(0, 2))
        padding = ~(valid & genes.ne(self.pad_id))
        padding[:, 0] = False
        hidden = self.backbone(genes, bins, padding, addition)
        return hidden[:, 0].float()

    def encode_cells(self, batch: Mapping[str, Tensor]) -> Tensor:
        genes, bins = batch["gene_ids"], batch["input_bins"]
        compartments, valid = batch["compartment_ids"], batch["token_valid_mask"].bool()
        if genes.ndim != 3 or any(t.shape != genes.shape for t in (bins, compartments, valid)):
            raise ValueError("Expected [B, cells, L] token arrays")
        batch_size, cells, length = genes.shape
        flat = [t.reshape(batch_size * cells, length) for t in (genes, bins, valid, compartments)]
        chunks = []
        for start in range(0, batch_size * cells, self.encoder_chunk_size):
            args = [t[start:start + self.encoder_chunk_size] for t in flat]
            if self.checkpointing and torch.is_grad_enabled():
                chunks.append(checkpoint(self._cls, *args, use_reentrant=False))
            else:
                chunks.append(self._cls(*args))
        return torch.cat(chunks).reshape(batch_size, cells, self.dim)

    def context(self, cells: Tensor, batch: Mapping[str, Tensor]) -> Tensor:
        nodes, coords = batch["node_indices"], batch["coords"].float()
        k = cells.shape[1] - 1
        if k < 1:
            return torch.zeros_like(cells[:, 0])
        indices = torch.arange(1, k + 1, device=cells.device).view(1, 1, k).expand(len(cells), 1, k)
        valid = (nodes[:, 1:].ge(0) & nodes[:, 1:].ne(nodes[:, :1])).unsqueeze(1)
        distance = (coords[:, 1:] - coords[:, :1]).norm(dim=-1).unsqueeze(1)
        lr = batch.get("lr_center")
        lr_bias = None
        if self.lr_enabled:
            if lr is None:
                raise ValueError("LR-enabled V9 requires lr_center")
            lr_bias = lr.float().unsqueeze(1)
        message = self.spatial(cells[:, :1], cells, (indices, valid, distance), lr_bias)
        return message[:, 0]

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False):
        cells = self.encode_cells(batch)
        z_self = cells[:, 0]
        z_ctx = self.context(cells, batch)
        embeddings = dict(self=z_self, ctx=z_ctx, joint=torch.cat((z_self, z_ctx), dim=-1))
        result = dict(embeddings=embeddings)
        if return_embeddings:
            return result
        compartments = batch["query_compartment_ids"]
        query = self.backbone.encoder(batch["query_gene_ids"]) + self.compartment_embedding(compartments)
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        steps = query.shape[1]
        result["self_logits"] = self.joint_decoder(query, z_self[:, None].expand(-1, steps, -1))
        result["ctx_logits"] = self.joint_decoder(query, z_ctx[:, None].expand(-1, steps, -1))
        result["lr_strength"] = self.spatial.lr_strength.detach()
        return result
