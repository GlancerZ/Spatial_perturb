"""Pretrained unordered gene sets, compartment readouts and local cytoplasm messaging.

Spatial neighborhoods are explicitly truncated to the supplied patch. Pure-niche
readouts see only independently encoded, pre-interaction neighbor cytoplasm.
Targets never enter this module; all four decoders predict nonzero/log-count.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .scgpt_backbone import ScGPTBackbone


def compartment_mean(hidden: Tensor, mask: Tensor, missing: Tensor) -> Tensor:
    weight = mask.unsqueeze(-1).to(hidden.dtype)
    count = weight.sum(-2)
    mean = (hidden * weight).sum(-2) / count.clamp_min(1)
    return torch.where(count > 0, mean, missing.to(hidden.dtype))


def patch_knn(coords: Tensor, node_indices: Tensor, sender_valid: Tensor, neighbors: int):
    """Return KNN within each patch; no self, duplicate sender IDs or hidden senders."""
    count = coords.shape[1]
    if count < 1 or neighbors < 1:
        raise ValueError('At least one patch cell and one neighbor slot are required')
    ids = node_indices
    equal = ids.unsqueeze(-1).eq(ids.unsqueeze(-2))
    prior = torch.ones(count, count, device=coords.device, dtype=torch.bool).tril(-1)
    first_occurrence = ~(equal & prior).any(-1)
    senders = sender_valid & first_occurrence & ids.ge(0)
    valid = ~equal & senders.unsqueeze(-2) & ids.ge(0).unsqueeze(-1)
    distance = torch.cdist(coords.float(), coords.float())
    score = distance.masked_fill(~valid, float('inf'))
    indices = score.argsort(dim=-1, stable=True)[..., :min(neighbors, count)]
    picked_valid = valid.gather(-1, indices)
    picked_distance = distance.gather(-1, indices)
    return indices, picked_valid, picked_distance


def gather_neighbors(values: Tensor, indices: Tensor) -> Tensor:
    batch, cells, dim = values.shape
    offset = torch.arange(batch, device=values.device).view(batch, 1, 1) * cells
    return values.reshape(batch * cells, dim)[indices + offset]


class LocalSpatialAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.distance_bias = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, heads))
        nn.init.zeros_(self.distance_bias[-1].weight)
        nn.init.zeros_(self.distance_bias[-1].bias)

    def forward(self, receiver: Tensor, senders: Tensor, graph) -> Tensor:
        indices, valid, distance = graph
        batch, cells, dim = receiver.shape
        k = indices.shape[-1]
        q = self.query(self.norm(receiver)).reshape(batch * cells, self.heads, 1, dim // self.heads)
        normalized = self.norm(senders)
        keys = gather_neighbors(self.key(normalized), indices)
        values = gather_neighbors(self.value(normalized), indices)
        keys = keys.reshape(batch * cells, k, self.heads, dim // self.heads).transpose(1, 2)
        values = values.reshape(batch * cells, k, self.heads, dim // self.heads).transpose(1, 2)
        # Give empty neighborhoods one dummy slot, then explicitly zero the result.
        any_neighbor = valid.any(-1)
        safe = valid.clone()
        safe[..., 0] |= ~any_neighbor
        bias = self.distance_bias(torch.log1p(distance).unsqueeze(-1).to(receiver.dtype))
        bias = bias.permute(0, 1, 3, 2).reshape(batch * cells, self.heads, 1, k)
        bias = bias.masked_fill(~safe.reshape(batch * cells, 1, 1, k), float('-inf'))
        message = F.scaled_dot_product_attention(q, keys, values, attn_mask=bias)
        message = self.out(message.reshape(batch, cells, dim))
        return message * any_neighbor.unsqueeze(-1).to(message.dtype)


class CytoplasmAdapter(nn.Module):
    def __init__(self, dim: int, heads: int, chunk_size: int, checkpointing: bool):
        super().__init__()
        self.spatial = LocalSpatialAttention(dim, heads)
        self.message = nn.Linear(dim, dim, bias=False)
        self.token_gate = nn.Linear(2 * dim, dim)
        self.receiver_gate = nn.Linear(2 * dim, dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.local = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0,
                                                batch_first=True, norm_first=True)
        self.chunk_size, self.checkpointing = chunk_size, checkpointing

    def forward(self, hidden: Tensor, receiver: Tensor, cyto_mask: Tensor, valid: Tensor, graph):
        message = self.message(self.spatial(receiver, receiver, graph))
        scale = self.residual_scale.tanh()
        receiver = receiver + scale * torch.sigmoid(self.receiver_gate(torch.cat([receiver, message], -1))) * message
        expanded = message.unsqueeze(-2).expand_as(hidden)
        gate = torch.sigmoid(self.token_gate(torch.cat([hidden, expanded], -1)))
        hidden = hidden + scale * gate * expanded * cyto_mask.unsqueeze(-1).to(hidden.dtype)
        shape = hidden.shape
        flat = hidden.flatten(0, 1)
        padding = ~valid.flatten(0, 1)
        chunks = []
        for start in range(0, len(flat), self.chunk_size):
            inputs, pad = flat[start:start + self.chunk_size], padding[start:start + self.chunk_size]
            if self.checkpointing and self.training and torch.is_grad_enabled():
                updated = checkpoint(self.local, inputs, src_key_padding_mask=pad, use_reentrant=False)
            else:
                updated = self.local(inputs, src_key_padding_mask=pad)
            chunks.append(updated)
        hidden = torch.cat(chunks).reshape(shape)
        # Empty cytoplasm keeps the receiver memory but never becomes a sender.
        receiver = compartment_mean(hidden, cyto_mask, receiver)
        return hidden, receiver


class Readout(nn.Module):
    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU(), nn.LayerNorm(dim))

    def forward(self, *values):
        return self.net(torch.cat(values, dim=-1))


class EmbeddingDecoder(nn.Module):
    """The only conditioning inputs are one cell embedding and query identities."""
    def __init__(self, dim: int):
        super().__init__()
        self.output = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, query: Tensor, embedding: Tensor) -> Tensor:
        return self.output(torch.cat([query, embedding.unsqueeze(1).expand_as(query)], -1))


class GeneDecoder(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, query: Tensor, center_hidden: Tensor, valid: Tensor):
        visible, _ = self.attention(query, center_hidden, center_hidden, key_padding_mask=~valid, need_weights=False)
        return self.output(torch.cat([query, visible], -1))


class SubcellularV2(nn.Module):
    def __init__(self, asset_dir: str | Path, adapter_layers: int = 2, neighbors: int = 16,
                 freeze_layers: int = 8, checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = 'scgpt_spatial', direct_niche_readout: bool = False,
                 backbone: nn.Module | None = None):
        super().__init__()
        if encoder_chunk_size < 1:
            raise ValueError('encoder_chunk_size must be positive')
        self.backbone_kind = backbone_kind
        if backbone is not None and getattr(backbone, 'backbone_kind', backbone_kind) != backbone_kind:
            raise ValueError('Injected backbone kind differs from the explicitly selected pretrained source')
        self.backbone = backbone if backbone is not None else ScGPTBackbone(
            asset_dir, checkpointing, backbone_kind=backbone_kind)
        self.vocab = self.backbone.vocab
        self.pad_id, self.dim = self.backbone.pad_id, self.backbone.dim
        self.freeze_layers, self.neighbors = freeze_layers, neighbors
        self.encoder_chunk_size = encoder_chunk_size
        self.checkpointing = checkpointing
        heads = self.backbone.nheads
        self.compartment_embedding = nn.Embedding(3, self.dim)
        nn.init.zeros_(self.compartment_embedding.weight)
        self.missing_nucleus = nn.Parameter(torch.randn(self.dim) * 0.02)
        self.missing_cytoplasm = nn.Parameter(torch.randn(self.dim) * 0.02)
        self.missing_niche = nn.Parameter(torch.randn(self.dim) * 0.02)
        self.niche_query = nn.Parameter(torch.randn(self.dim) * 0.02)
        self.intrinsic_readout = Readout(2 * self.dim, self.dim)
        self.joint_readout = Readout(3 * self.dim, self.dim)
        self.niche_attention = LocalSpatialAttention(self.dim, heads)
        self.niche_readout = nn.LayerNorm(self.dim)
        self.adapters = nn.ModuleList([CytoplasmAdapter(self.dim, heads, encoder_chunk_size, checkpointing)
                                       for _ in range(adapter_layers)])
        self.gene_decoder = GeneDecoder(self.dim, heads)
        self.repr_decoder = EmbeddingDecoder(self.dim)
        self.neighbor_decoder = EmbeddingDecoder(self.dim)
        self.direct_niche_readout = bool(direct_niche_readout)
        if self.direct_niche_readout:
            self.niche_to_joint = nn.Linear(self.dim, self.dim, bias=False)
            nn.init.zeros_(self.niche_to_joint.weight)
        self.set_backbone_trainable(full=False)

    def set_backbone_trainable(self, full: bool):
        self.backbone.requires_grad_(full)
        if not full:
            for layer in self.backbone.transformer_encoder.layers[self.freeze_layers:]:
                layer.requires_grad_(True)

    def _encode(self, batch: Mapping[str, Tensor]) -> Tensor:
        genes, bins = batch['gene_ids'], batch['input_bins']
        compartment = batch['compartment_ids']
        valid = batch['token_valid_mask']
        if not valid[..., 0].all():
            raise ValueError('Each cell must have a valid non-gene CLS/sentinel at position 0')
        if not compartment[..., 0].eq(2).all():
            raise ValueError('CLS/sentinel must have compartment ID 2')
        shape = genes.shape
        genes, bins, compartment, valid = [value.flatten(0, 1) for value in (genes, bins, compartment, valid)]
        chunks = []
        for start in range(0, len(genes), self.encoder_chunk_size):
            end = start + self.encoder_chunk_size
            comp = self.compartment_embedding(compartment[start:end])
            hidden = self.backbone(genes[start:end], bins[start:end], ~valid[start:end], comp)
            chunks.append(hidden)
        return torch.cat(chunks).reshape(*shape, self.dim)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False):
        hidden0 = self._encode(batch)
        valid = batch['token_valid_mask']
        nmask = valid & batch['compartment_ids'].eq(0)
        cmask = valid & batch['compartment_ids'].eq(1)
        n0 = compartment_mean(hidden0, nmask, self.missing_nucleus)
        c0 = compartment_mean(hidden0, cmask, self.missing_cytoplasm)
        # Preserve a direct path from pretrained features at initialization;
        # small learned residuals can specialize the compartment combination.
        intrinsic = 0.5 * (n0 + c0) + 0.1 * self.intrinsic_readout(n0, c0)
        graph = patch_knn(batch['coords'], batch['node_indices'], cmask.any(-1), self.neighbors)
        niche = self.niche_attention(self.niche_query.expand_as(c0), c0, graph)
        niche = torch.where(graph[1].any(-1, keepdim=True), niche, self.missing_niche.to(niche.dtype))
        niche = self.niche_readout(niche)
        hidden, receiver = hidden0, c0
        for adapter in self.adapters:
            hidden, receiver = adapter(hidden, receiver, cmask, valid, graph)
        n_post = compartment_mean(hidden, nmask, self.missing_nucleus)
        joint = intrinsic + 0.1 * self.joint_readout(intrinsic, n_post, receiver)
        if self.direct_niche_readout:
            joint = joint + self.niche_to_joint(niche)
        result = {'embeddings': {'intrinsic': intrinsic, 'joint': joint, 'niche': niche}}
        if return_embeddings:
            return result
        query = (self.backbone.encoder(batch['query_gene_ids']) +
                 self.compartment_embedding(batch['query_compartment_ids']))
        result.update(gene_pred=self.gene_decoder(query, hidden0[:, 0], valid[:, 0]),
                      intrinsic_pred=self.repr_decoder(query, intrinsic[:, 0]),
                      joint_pred=self.repr_decoder(query, joint[:, 0]),
                      neighbor_pred=self.neighbor_decoder(query, niche[:, 0]))
        return result
