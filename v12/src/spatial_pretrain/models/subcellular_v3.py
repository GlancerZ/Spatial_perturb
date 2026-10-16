"""Shared ordinary-scGPT compartment CLSs with local/spatial residual exchange.

Each compartment is encoded as an independent sequence with the original CLS.
Only cytoplasm states cross cell boundaries; nuclear states can receive spatial
information indirectly through within-cell exchange. Targets are not inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .scgpt_backbone import ScGPTBackbone
from .subcellular_v2 import LocalSpatialAttention, patch_knn


class WithinCellExchange(nn.Module):
    """Pre-norm two-token self-attention and FFN; exactly identity at init."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, states: Tensor) -> Tensor:
        shape = states.shape
        pair = states.reshape(-1, 2, shape[-1])
        normalized = self.attention_norm(pair)
        message, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        pair = pair + message
        pair = pair + self.ffn(self.ffn_norm(pair))
        return pair.reshape(shape)


class CytoplasmExchange(nn.Module):
    """Cytoplasm-only Q/K/V, with an initially zero spatial residual."""
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention = LocalSpatialAttention(dim, heads)
        nn.init.zeros_(self.attention.out.weight)
        nn.init.zeros_(self.attention.out.bias)

    def forward(self, states: Tensor, graph) -> Tensor:
        cytoplasm = states[..., 1, :]
        message = self.attention(cytoplasm, cytoplasm, graph)
        return torch.stack((states[..., 0, :], cytoplasm + message), dim=-2)


class CLSQueryDecoder(nn.Module):
    """Predict nonzero and relative-rank logits using only CLS and query IDs."""
    def __init__(self, dim: int, context_dim: int):
        super().__init__()
        self.output = nn.Sequential(nn.LayerNorm(dim + context_dim),
                                    nn.Linear(dim + context_dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        return self.output(torch.cat((query, context), dim=-1))


class SubcellularV3(nn.Module):
    def __init__(self, asset_dir: str | Path, neighbors: int = 16,
                 checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = 'scgpt', backbone: nn.Module | None = None):
        super().__init__()
        if backbone_kind != 'scgpt':
            raise ValueError('V3 only supports the explicitly verified ordinary scGPT backbone')
        if encoder_chunk_size < 1 or neighbors < 1:
            raise ValueError('encoder_chunk_size and neighbors must be positive')
        if backbone is not None and getattr(backbone, 'backbone_kind', 'scgpt') != 'scgpt':
            raise ValueError('Injected V3 backbone must be ordinary scGPT')
        self.backbone_kind = backbone_kind
        self.backbone = backbone if backbone is not None else ScGPTBackbone(
            asset_dir, checkpointing=checkpointing, backbone_kind='scgpt')
        self.vocab = self.backbone.vocab
        self.pad_id, self.cls_id, self.dim = self.backbone.pad_id, self.vocab['<cls>'], self.backbone.dim
        self.pad_value = float(getattr(self.backbone, 'config', {}).get('pad_value', -2.0))
        self.neighbors, self.encoder_chunk_size = int(neighbors), int(encoder_chunk_size)
        self.checkpointing = bool(checkpointing)
        self.compartment_embedding = nn.Embedding(2, self.dim)
        nn.init.zeros_(self.compartment_embedding.weight)
        self.local_blocks = nn.ModuleList([WithinCellExchange(self.dim, self.backbone.nheads) for _ in range(3)])
        self.spatial_blocks = nn.ModuleList([CytoplasmExchange(self.dim, self.backbone.nheads) for _ in range(2)])
        self.compartment_decoder = CLSQueryDecoder(self.dim, self.dim)
        self.joint_decoder = CLSQueryDecoder(self.dim, 2 * self.dim)
        self.set_backbone_trainable(full=False)

    def set_backbone_trainable(self, full: bool):
        """Freeze all backbone weights, or explicitly open the whole backbone."""
        self.backbone.requires_grad_(bool(full))

    def set_backbone_trainable_layers(self, count: int):
        """Open only the final K transformer layers; keep input embeddings frozen."""
        layers = self.backbone.transformer_encoder.layers
        if not isinstance(count, int) or not 0 <= count <= len(layers):
            raise ValueError(f'count must be an integer between 0 and {len(layers)}')
        self.backbone.requires_grad_(False)
        if count:
            for layer in layers[-count:]:
                layer.requires_grad_(True)

    def _split_sequences(self, batch: Mapping[str, Tensor]):
        """Pack N/C independently, preserving all valid genes and a CLS each.

        Rows are interleaved cell0-N, cell0-C, cell1-N, cell1-C. Compartment
        tokens never share an encoder sequence, including when chunks mix rows.
        """
        genes, bins = batch['gene_ids'], batch['input_bins']
        compartments, valid = batch['compartment_ids'], batch['token_valid_mask'].bool()
        if genes.ndim != 3 or any(value.shape != genes.shape for value in (bins, compartments, valid)):
            raise ValueError('Packed gene/bin/compartment/valid arrays must have equal [B,C,L] shapes')
        if not bool(valid[..., 0].all()) or not bool(compartments[..., 0].eq(2).all()):
            raise ValueError('Each packed cell must retain a valid non-gene CLS at slot zero')
        batch_size, cells, _ = genes.shape
        flat_genes, flat_bins = genes.flatten(0, 1), bins.flatten(0, 1)
        flat_compartments, flat_valid = compartments.flatten(0, 1), valid.flatten(0, 1)
        masks = (flat_valid[:, None, :] & flat_compartments[:, None, :].eq(
            torch.arange(2, device=genes.device).view(1, 2, 1)))
        counts = masks.sum(-1)
        length = 1 + int(counts.max().item())
        rows = batch_size * cells * 2
        split_genes = genes.new_full((rows, length), self.pad_id)
        split_bins = bins.new_full((rows, length), self.pad_value)
        split_valid = valid.new_zeros((rows, length))
        split_genes[:, 0] = self.cls_id
        split_valid[:, 0] = True
        cell_index, compartment_index, token_index = masks.nonzero(as_tuple=True)
        destinations = masks.long().cumsum(-1)[cell_index, compartment_index, token_index]
        row_index = 2 * cell_index + compartment_index
        split_genes[row_index, destinations] = flat_genes[cell_index, token_index]
        split_bins[row_index, destinations] = flat_bins[cell_index, token_index]
        split_valid[row_index, destinations] = True
        types = torch.arange(2, device=genes.device).repeat(batch_size * cells)
        return split_genes, split_bins, split_valid, types, counts.reshape(batch_size, cells, 2).gt(0)

    def encode_compartments(self, batch: Mapping[str, Tensor]):
        genes, bins, valid, types, visible = self._split_sequences(batch)
        cls_states = []
        for start in range(0, len(genes), self.encoder_chunk_size):
            end = start + self.encoder_chunk_size
            # No no_grad/detach: frozen pretrained weights must still transmit
            # gradients to the trainable compartment-type input.
            addition = self.compartment_embedding(types[start:end]).unsqueeze(1)
            arguments = (genes[start:end], bins[start:end], ~valid[start:end], addition)
            if self.checkpointing and torch.is_grad_enabled():
                # Frozen backbone runs in eval mode to disable pretrained
                # dropout, but its input type vectors still require gradients.
                # Its own per-layer checkpointing is gated on training mode;
                # this outer checkpoint also protects the frozen/eval stage.
                cls_state = checkpoint(self._backbone_cls, *arguments, use_reentrant=False)
            else:
                cls_state = self._backbone_cls(*arguments)
            cls_states.append(cls_state)
        initial = torch.cat(cls_states).reshape(*visible.shape, self.dim)
        visible = visible & batch['node_indices'].ge(0).unsqueeze(-1)
        return initial, visible

    def _backbone_cls(self, genes, bins, padding, addition):
        # Do not retain a strided view into the complete token-output storage.
        return self.backbone(genes, bins, padding, addition)[:, 0].contiguous()

    def interact(self, initial: Tensor, graph, *, spatial: bool) -> Tensor:
        states = initial
        for index, local in enumerate(self.local_blocks):
            states = local(states)
            if spatial and index < len(self.spatial_blocks):
                states = self.spatial_blocks[index](states, graph)
        return states

    @staticmethod
    def normalized_states(states: Tensor, node_indices: Tensor) -> Tensor:
        normalized = F.normalize(states.float(), dim=-1, eps=1e-8)
        return normalized * node_indices.ge(0)[..., None, None].to(normalized.dtype)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False):
        initial, visible = self.encode_compartments(batch)
        # Original visibility is fixed across both spatial stages: locally
        # imputed cytoplasm cannot turn a hidden/empty cell into a sender.
        graph = patch_knn(batch['coords'], batch['node_indices'], visible[..., 1], self.neighbors)
        final = self.interact(initial, graph, spatial=True)
        base_states = self.normalized_states(initial, batch['node_indices'])
        joint_states = self.normalized_states(final, batch['node_indices'])
        embeddings = dict(base=base_states.flatten(-2), joint=joint_states.flatten(-2),
                          nucleus=joint_states[..., 0, :], cytoplasm=joint_states[..., 1, :])
        result = dict(embeddings=embeddings, compartment_visible_mask=visible)
        if return_embeddings:
            intrinsic = self.interact(initial, graph, spatial=False)
            embeddings['intrinsic'] = self.normalized_states(intrinsic, batch['node_indices']).flatten(-2)
            return result
        query_compartments = batch['query_compartment_ids']
        if bool(((query_compartments < 0) | (query_compartments > 1)).any()):
            raise ValueError('Query compartments must be nuclear=0 or cytoplasm=1')
        query = (self.backbone.encoder(batch['query_gene_ids'])
                 + self.compartment_embedding(query_compartments))
        # Official gene embeddings are LayerNorm-scaled (norm ~sqrt(dim)),
        # while each context CLS has norm 1. Match block scales before the
        # decoder's joint LayerNorm so query identity does not swamp context.
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        selected_initial = base_states[:, 0].gather(
            1, query_compartments[..., None].expand(-1, -1, self.dim))
        joint_context = embeddings['joint'][:, 0, None, :].expand(-1, query.shape[1], -1)
        result['compartment_pred'] = self.compartment_decoder(query, selected_initial)
        result['joint_pred'] = self.joint_decoder(query, joint_context)
        return result
