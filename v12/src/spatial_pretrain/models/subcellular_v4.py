"""Unordered gene sets, cytoplasm communication and within-cell feedback.

Each compartment produces learned slots from gene hidden states. The sequence
is Local_pre -> Spatial -> Local_feedback; only cytoplasm crosses cells. One
decoder reads exactly the normalized joint embedding used by downstream probes.
"""
from __future__ import annotations

from math import gcd
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .scgpt_backbone import ScGPTBackbone
from .subcellular_v2 import LocalSpatialAttention, patch_knn


class GeneSlotPooling(nn.Module):
    """Shared query identities pool only gene keys, with explicit empty states."""

    def __init__(self, input_dim: int, slots: int, dim: int, heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, dim) * .02)
        self.missing = nn.Parameter(torch.randn(2, slots, dim) * .02)
        self.gene_projection = nn.Linear(input_dim, dim)
        self.gene_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)

    def forward(self, hidden: Tensor, gene_valid: Tensor, compartments: Tensor) -> Tensor:
        # Never run softmax over an entirely masked sequence. Missing states
        # retain learned slot identities and compartment-specific parameters.
        queries = self.queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        slots = queries + self.missing[compartments]
        live = gene_valid.any(-1).nonzero(as_tuple=True)[0]
        if live.numel():
            genes = self.gene_norm(self.gene_projection(hidden[live]))
            message, _ = self.attention(queries[live], genes, genes,
                                        key_padding_mask=~gene_valid[live], need_weights=False)
            slots = slots.index_copy(0, live, queries[live] + message)
        return slots


class WithinCellSlotExchange(nn.Module):
    """Self-attention among this cell's 2*K slots; identity at initialization."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, states: Tensor) -> Tensor:
        shape = states.shape
        sequence = states.reshape(-1, shape[-3] * shape[-2], shape[-1])
        normalized = self.attention_norm(sequence)
        message, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        sequence = sequence + message
        sequence = sequence + self.ffn(self.ffn_norm(sequence))
        return sequence.reshape(shape)


class CytoplasmSlotExchange(nn.Module):
    """Flatten all cytoplasm slots, exchange across cells, then restore slots."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.attention = LocalSpatialAttention(dim, heads)
        nn.init.zeros_(self.attention.out.weight)
        nn.init.zeros_(self.attention.out.bias)

    def forward(self, states: Tensor, graph) -> Tensor:
        cytoplasm = states[..., 1, :, :]
        flattened = cytoplasm.flatten(-2)
        message = self.attention(flattened, flattened, graph).reshape_as(cytoplasm)
        return torch.stack((states[..., 0, :, :], cytoplasm + message), dim=-3)


class JointBinDecoder(nn.Module):
    """One head, conditioned only on gene/compartment identity and joint z."""

    def __init__(self, query_dim: int, context_dim: int, num_bins: int):
        super().__init__()
        self.output = nn.Sequential(nn.LayerNorm(query_dim + context_dim),
                                    nn.Linear(query_dim + context_dim, query_dim),
                                    nn.GELU(), nn.Linear(query_dim, num_bins))

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        return self.output(torch.cat((query, context), dim=-1))


class SubcellularV4(nn.Module):
    """Ordinary scGPT gene slots with one final 51-bin reconstruction head."""

    num_bins = 51

    def __init__(self, asset_dir: str | Path, neighbors: int = 16,
                 checkpointing: bool = True, encoder_chunk_size: int = 32,
                 *, backbone_kind: str = 'scgpt', backbone: nn.Module | None = None,
                 slots_per_compartment: int = 8, slot_dim: int = 64,
                 local_pre_layers: int = 1, spatial_layers: int = 2,
                 local_post_layers: int = 2):
        super().__init__()
        if backbone_kind != 'scgpt':
            raise ValueError('V4 only supports the explicitly verified ordinary scGPT backbone')
        if backbone is not None and getattr(backbone, 'backbone_kind', 'scgpt') != 'scgpt':
            raise ValueError('Injected V4 backbone must be ordinary scGPT')
        positive = dict(neighbors=neighbors, encoder_chunk_size=encoder_chunk_size,
                        slots_per_compartment=slots_per_compartment, slot_dim=slot_dim)
        for name, value in positive.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name, value in dict(local_pre_layers=local_pre_layers, spatial_layers=spatial_layers,
                                local_post_layers=local_post_layers).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        self.backbone_kind = backbone_kind
        self.backbone = backbone if backbone is not None else ScGPTBackbone(
            asset_dir, checkpointing=checkpointing, backbone_kind='scgpt')
        self.vocab = self.backbone.vocab
        self.pad_id, self.cls_id = self.backbone.pad_id, self.vocab['<cls>']
        self.pad_value = float(getattr(self.backbone, 'config', {}).get('pad_value', -2.))
        self.slots_per_compartment, self.slot_dim = slots_per_compartment, slot_dim
        self.dim = slots_per_compartment * slot_dim
        self.neighbors, self.encoder_chunk_size = neighbors, encoder_chunk_size
        self.checkpointing = bool(checkpointing)
        slot_heads = gcd(self.backbone.nheads, slot_dim)
        spatial_heads = gcd(self.backbone.nheads, self.dim)
        self.arch_config = dict(model_type='subcellular_geneset_v4', backbone_kind=backbone_kind,
                                backbone_dim=self.backbone.dim, slots_per_compartment=slots_per_compartment,
                                slot_dim=slot_dim, local_pre_layers=local_pre_layers,
                                spatial_layers=spatial_layers, local_post_layers=local_post_layers,
                                neighbors=neighbors, num_bins=self.num_bins,
                                pooling_heads=slot_heads, spatial_heads=spatial_heads)
        self.compartment_embedding = nn.Embedding(2, self.backbone.dim)
        nn.init.zeros_(self.compartment_embedding.weight)
        self.slot_pool = GeneSlotPooling(self.backbone.dim, slots_per_compartment, slot_dim, slot_heads)
        self.local_pre_blocks = nn.ModuleList([
            WithinCellSlotExchange(slot_dim, slot_heads) for _ in range(local_pre_layers)])
        self.spatial_blocks = nn.ModuleList([
            CytoplasmSlotExchange(self.dim, spatial_heads) for _ in range(spatial_layers)])
        self.local_post_blocks = nn.ModuleList([
            WithinCellSlotExchange(slot_dim, slot_heads) for _ in range(local_post_layers)])
        self.joint_decoder = JointBinDecoder(self.backbone.dim, 2 * self.dim, self.num_bins)
        self.set_backbone_trainable(full=False)

    def train(self, mode: bool = True):
        super().train(mode)
        if not any(parameter.requires_grad for parameter in self.backbone.parameters()):
            self.backbone.eval()
        return self

    def set_backbone_trainable(self, full: bool):
        self.backbone.requires_grad_(bool(full))
        self.backbone.train(self.training and bool(full))

    def set_backbone_trainable_layers(self, count: int):
        layers = self.backbone.transformer_encoder.layers
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(layers):
            raise ValueError(f'count must be an integer between 0 and {len(layers)}')
        self.backbone.requires_grad_(False)
        if count:
            for layer in layers[-count:]:
                layer.requires_grad_(True)
        self.backbone.train(self.training and bool(count))

    def _split_sequences(self, batch: Mapping[str, Tensor]):
        """Pack independent N/C sequences, preserving CLS for the backbone only."""
        genes, bins = batch['gene_ids'], batch['input_bins']
        compartments, valid = batch['compartment_ids'], batch['token_valid_mask'].bool()
        if genes.ndim != 3 or any(value.shape != genes.shape for value in (bins, compartments, valid)):
            raise ValueError('Packed gene/bin/compartment/valid arrays must have equal [B,C,L] shapes')
        if min(genes.shape) < 1:
            raise ValueError('Packed input must contain at least one cell and CLS')
        if (not bool(valid[..., 0].all()) or not bool(compartments[..., 0].eq(2).all())
                or not bool(genes[..., 0].eq(self.cls_id).all())):
            raise ValueError('Each packed cell must retain a valid non-gene CLS at slot zero')
        batch_size, cells, _ = genes.shape
        flat_genes, flat_bins = genes.flatten(0, 1), bins.flatten(0, 1)
        flat_compartments = compartments.flatten(0, 1)
        gene_valid = (valid & genes.ne(self.pad_id) & genes.ne(self.cls_id)).flatten(0, 1)
        masks = gene_valid[:, None, :] & flat_compartments[:, None, :].eq(
            torch.arange(2, device=genes.device).view(1, 2, 1))
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

    def _backbone_slots(self, genes: Tensor, bins: Tensor, valid: Tensor, types: Tensor) -> Tensor:
        addition = self.compartment_embedding(types).unsqueeze(1)
        hidden = self.backbone(genes, bins, ~valid, addition)
        gene_valid = valid & genes.ne(self.cls_id) & genes.ne(self.pad_id)
        return self.slot_pool(hidden, gene_valid, types)

    def encode_compartments(self, batch: Mapping[str, Tensor]):
        genes, bins, valid, types, visible = self._split_sequences(batch)
        chunks = []
        for start in range(0, len(genes), self.encoder_chunk_size):
            end = start + self.encoder_chunk_size
            arguments = (genes[start:end], bins[start:end], valid[start:end], types[start:end])
            # Pool INSIDE the checkpointed chunk, so retained outputs are K*d
            # slots instead of every gene hidden state in the complete patch.
            # Frozen backbone weights still transmit gradients to input types.
            if self.checkpointing and torch.is_grad_enabled():
                chunk = checkpoint(self._backbone_slots, *arguments, use_reentrant=False)
            else:
                chunk = self._backbone_slots(*arguments)
            chunks.append(chunk)
        initial = torch.cat(chunks).reshape(*visible.shape, self.slots_per_compartment, self.slot_dim)
        visible = visible & batch['node_indices'].ge(0).unsqueeze(-1)
        return initial, visible

    def _after_local(self, local: Tensor, graph, *, spatial: bool) -> Tensor:
        states = local
        if spatial:
            for block in self.spatial_blocks:
                states = block(states, graph)
        for block in self.local_post_blocks:
            states = block(states)
        return states

    def interact(self, initial: Tensor, graph, *, spatial: bool = True) -> Tensor:
        states = initial
        for block in self.local_pre_blocks:
            states = block(states)
        return self._after_local(states, graph, spatial=spatial)

    @staticmethod
    def normalized_states(states: Tensor, node_indices: Tensor) -> Tensor:
        # Normalize each entire compartment, not each slot independently.
        normalized = F.normalize(states.flatten(-2).float(), dim=-1, eps=1e-8)
        return normalized * node_indices.ge(0)[..., None, None].to(normalized.dtype)

    def forward(self, batch: Mapping[str, Tensor], return_embeddings: bool = False):
        initial, visible = self.encode_compartments(batch)
        # Eligibility is measured before imputation and is reused unchanged.
        graph = patch_knn(batch['coords'], batch['node_indices'], visible[..., 1], self.neighbors)
        local = initial
        for block in self.local_pre_blocks:
            local = block(local)
        final = self._after_local(local, graph, spatial=True)
        base_states = self.normalized_states(initial, batch['node_indices'])
        local_states = self.normalized_states(local, batch['node_indices'])
        joint_states = self.normalized_states(final, batch['node_indices'])
        embeddings = dict(base=base_states.flatten(-2), joint=joint_states.flatten(-2),
                          nucleus=joint_states[..., 0, :], cytoplasm=joint_states[..., 1, :],
                          nucleus_local=local_states[..., 0, :], cytoplasm_local=local_states[..., 1, :])
        result = dict(embeddings=embeddings, compartment_visible_mask=visible)
        if return_embeddings:
            intrinsic = self._after_local(local, graph, spatial=False)
            embeddings['intrinsic'] = self.normalized_states(intrinsic, batch['node_indices']).flatten(-2)
            return result
        query_compartments = batch['query_compartment_ids']
        if bool(((query_compartments < 0) | (query_compartments > 1)).any()):
            raise ValueError('Query compartments must be nuclear=0 or cytoplasm=1')
        query = (self.backbone.encoder(batch['query_gene_ids'])
                 + self.compartment_embedding(query_compartments))
        query = F.normalize(query.float(), dim=-1, eps=1e-8)
        context = embeddings['joint'][:, 0, None, :].expand(-1, query.shape[1], -1)
        result['joint_logits'] = self.joint_decoder(query, context)
        return result
