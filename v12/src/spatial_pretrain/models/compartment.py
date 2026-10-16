"""Within-cell token attention and cytoplasm-restricted cross-cell attention."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import sdpa_kernel, SDPBackend

from spatial_pretrain.models.spatial_bias import RelativeCellSpatialBias
from spatial_pretrain.models.ssgt import FeedForward, GeneEmbeddingStem, SDPAAttention


class CytoplasmicCrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, require_flash: bool = False) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by cell_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.require_flash = require_flash
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def _attend(self, query, keys, values, key_indices):
        # Gather inside checkpoint so each receiving cell does not retain a full
        # copy of its neighbors' K/V for backward. Every query sees all allowed
        # keys in a single softmax; no independent per-neighbor normalization.
        key = keys[key_indices].transpose(0, 1).unsqueeze(0)
        value = values[key_indices].transpose(0, 1).unsqueeze(0)
        query = query.transpose(0, 1).unsqueeze(0)
        kernel = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.require_flash and query.is_cuda else nullcontext()
        with kernel:
            context = F.scaled_dot_product_attention(
                query, key, value, dropout_p=self.dropout if self.training else 0.0,
            )
        return context.squeeze(0).transpose(0, 1).flatten(1)

    def forward(self, tokens, compartment_ids, valid, node_indices):
        batch, cells, length, dim = tokens.shape
        cyto = valid & compartment_ids.eq(1)
        # Standard learned projections at token width; no cell pooling/flattening.
        shape = (batch, cells, length, self.heads, self.head_dim)
        query = self.q_proj(tokens).view(shape)
        keys = self.k_proj(tokens).view(shape)
        values = self.v_proj(tokens).view(shape)
        outputs = []
        active_batches = []
        for b in range(batch):
            # Only the first copy of an original cell contributes memory.
            cell_ids = node_indices[b]
            first = torch.ones(cells, dtype=torch.bool, device=tokens.device)
            for c in range(1, cells):
                first[c] = ~cell_ids[:c].eq(cell_ids[c]).any()
            memory_valid = cyto[b] & first[:, None]
            key_table = keys[b].flatten(0, 1)
            value_table = values[b].flatten(0, 1)
            cell_outputs, cell_active = [], []
            for c in range(cells):
                q_indices = cyto[b, c].nonzero(as_tuple=True)[0]
                allowed = memory_valid & cell_ids.ne(cell_ids[c])[:, None]
                k_indices = allowed.flatten().nonzero(as_tuple=True)[0]
                context = query.new_zeros((length, dim))
                active = torch.zeros(length, dtype=torch.bool, device=tokens.device)
                if q_indices.numel() and k_indices.numel():
                    args = (query[b, c, q_indices], key_table, value_table, k_indices)
                    if self.training and torch.is_grad_enabled():
                        update = checkpoint(self._attend, *args, use_reentrant=False)
                    else:
                        update = self._attend(*args)
                    context = context.index_copy(0, q_indices, update)
                    active[q_indices] = True
                cell_outputs.append(context)
                cell_active.append(active)
            outputs.append(torch.stack(cell_outputs))
            active_batches.append(torch.stack(cell_active))
        active = torch.stack(active_batches)
        context = torch.stack(outputs)
        # Keep empty-cell/rank paths in the autograd graph with exactly zero
        # contribution, including Q/K/V parameters (needed by DDP).
        context = context + (query.sum() + keys.sum() + values.sum()) * 0.0
        output = self.out_dropout(self.out_proj(context))
        return output.masked_fill(~active.unsqueeze(-1), 0.0), active


class StackCytoplasmicCellAttention(nn.Module):
    """Flatten each cytoplasmic table, then apply dense standard cell Q/K/V.

    Follows the attention axes in ArcInstitute/stack TabularAttentionLayer.
    A token-width adapter keeps the full flattened projections tractable while
    retaining the existing wider intracellular/GenePT representation.
    """

    def __init__(self, dim, heads, dropout, nuclear_tokens, cytoplasmic_tokens,
                 token_dim=16, require_flash=False):
        super().__init__()
        self.nuclear_tokens = nuclear_tokens
        self.cytoplasmic_tokens = cytoplasmic_tokens
        self.token_dim = token_dim
        self.require_flash = require_flash
        self.flat_dim = cytoplasmic_tokens * token_dim
        self.heads = heads
        if min(nuclear_tokens, cytoplasmic_tokens, token_dim, heads) <= 0:
            raise ValueError("Compartment budgets, token width and heads must be positive")
        if self.flat_dim % heads:
            raise ValueError("cytoplasmic_tokens * cell_token_dim must be divisible by cell_heads")
        self.head_dim = self.flat_dim // heads
        self.dropout = dropout
        self.token_in = nn.Linear(dim, token_dim)
        self.qkv = nn.Linear(self.flat_dim, 3 * self.flat_dim, bias=False)
        self.proj = nn.Linear(self.flat_dim, self.flat_dim)
        self.token_out = nn.Linear(token_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def _flash_group(self, query, key, value, rows, columns):
        q = query.transpose(0, 1)[rows].unsqueeze(2)
        k = key.transpose(0, 1)[columns].permute(0, 2, 1, 3)
        v = value.transpose(0, 1)[columns].permute(0, 2, 1, 3)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0,
            ).squeeze(2)

    def _flash_cells(self, query, key, value, allowed):
        # Remove prohibited keys before FlashAttention. Group equally sized
        # memory sets; checkpoint small gathers to avoid saving C copies of K/V.
        batches = []
        for b in range(query.shape[0]):
            counts = allowed[b].sum(-1)
            rows_out, values_out = [], []
            for count in counts.unique().tolist():
                if count == 0:
                    continue
                rows = counts.eq(count).nonzero(as_tuple=True)[0]
                columns = allowed[b, rows].nonzero(as_tuple=True)[1].reshape(len(rows), count)
                for start in range(0, len(rows), 8):
                    indices, memory = rows[start:start+8], columns[start:start+8]
                    args = (query[b], key[b], value[b], indices, memory)
                    attended = (checkpoint(self._flash_group, *args, use_reentrant=False)
                                if self.training and torch.is_grad_enabled() else self._flash_group(*args))
                    rows_out.append(indices)
                    values_out.append(attended)
            output = query.new_zeros(query.shape[2], self.heads, self.head_dim)
            if rows_out:
                output = output.index_copy(0, torch.cat(rows_out), torch.cat(values_out))
            batches.append(output.transpose(0, 1))
        return torch.stack(batches) + (query.sum() + key.sum() + value.sum()) * 0.0

    def forward(self, tokens, compartment_ids, valid, node_indices):
        batch, cells, length, dim = tokens.shape
        if length != self.nuclear_tokens + self.cytoplasmic_tokens:
            raise ValueError("Token length must match the two fixed compartment budgets")
        start = self.nuclear_tokens
        cyto = valid[:, :, start:] & compartment_ids[:, :, start:].eq(1)
        # Mask after the biased adapter too: PAD must contribute no input features.
        cytoplasm = self.token_in(tokens[:, :, start:]).masked_fill(~cyto.unsqueeze(-1), 0.0)
        cell_vectors = cytoplasm.flatten(2)  # [B, C, T_cyto * d_cyto]
        qkv = self.qkv(cell_vectors).reshape(batch, cells, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)

        same_cell = node_indices[:, :, None].eq(node_indices[:, None, :])
        earlier = torch.ones(cells, cells, dtype=torch.bool, device=tokens.device).tril(-1)
        first_copy = ~(same_cell & earlier).any(-1)
        has_cyto = cyto.any(-1)
        allowed = (~same_cell & has_cyto[:, :, None]
                   & (has_cyto & first_copy)[:, None, :])
        active_cells = allowed.any(-1)
        # Give inactive rows a harmless temporary key, then discard their output.
        # This avoids all-masked softmax rows even in the empty-memory case.
        fallback = torch.arange(cells, device=tokens.device).eq(0)
        safe_allowed = allowed | (~active_cells[:, :, None] & fallback)
        if self.require_flash and query.is_cuda:
            context = self._flash_cells(query, key, value, allowed)
        else:
            context = F.scaled_dot_product_attention(
                query, key, value, attn_mask=safe_allowed[:, None],
                dropout_p=self.dropout if self.training else 0.0,
            )
        context = context.transpose(1, 2).reshape(batch, cells, self.flat_dim)
        context = self.proj(context).reshape(batch, cells, self.cytoplasmic_tokens, self.token_dim)
        update = self.out_dropout(self.token_out(context))
        active_cyto = cyto & active_cells[:, :, None]
        update = update.masked_fill(~active_cyto.unsqueeze(-1), 0.0)
        # Nuclear states bypass both attention and the subsequent cross-cell FFN.
        prefix = update.new_zeros(batch, cells, start, dim)
        active_prefix = torch.zeros(batch, cells, start, dtype=torch.bool, device=tokens.device)
        return torch.cat((prefix, update), dim=2), torch.cat((active_prefix, active_cyto), dim=2)


class CytoplasmicMeanCellAttention(nn.Module):
    """One masked-mean cytoplasmic vector per cell; broadcast the neighbor update.

    The shared Flash helpers operate only on [B,H,C,D] tensors and allowed
    cell pairs, so their behavior is independent of how a cell is represented.
    """
    _flash_group = StackCytoplasmicCellAttention._flash_group
    _flash_cells = StackCytoplasmicCellAttention._flash_cells

    def __init__(self, dim, heads, dropout, require_flash=False, spatial_bias=False,
                 angle_bins=16, distance_edges=(10., 20., 40., 80., 160., 320.), coordinate_scale=1.):
        super().__init__()
        if dim % heads:
            raise ValueError("hidden_dim must be divisible by cell_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.dropout, self.require_flash = dropout, require_flash
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)
        self.spatial_bias = RelativeCellSpatialBias(heads, angle_bins, distance_edges, coordinate_scale) if spatial_bias else None

    def forward(self, tokens, compartment_ids, valid, node_indices, coords=None):
        batch, cells, _, dim = tokens.shape
        cyto = valid & compartment_ids.eq(1)
        counts = cyto.sum(-1)
        pooled = tokens.masked_fill(~cyto.unsqueeze(-1), 0).float().sum(2)
        pooled = (pooled / counts.clamp_min(1).unsqueeze(-1)).to(tokens.dtype)
        qkv = self.qkv(pooled).reshape(batch, cells, 3, self.heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        same = node_indices[:, :, None].eq(node_indices[:, None, :])
        earlier = torch.ones(cells, cells, dtype=torch.bool, device=tokens.device).tril(-1)
        first = ~(same & earlier).any(-1)
        has_cyto = counts.gt(0)
        allowed = ~same & has_cyto[:, :, None] & (has_cyto & first)[:, None, :]
        active_cells = allowed.any(-1)
        if self.spatial_bias is not None:
            if coords is None or coords.shape != (batch, cells, 2):
                raise ValueError('Coordinates are required when cell_spatial_bias is enabled')
            fallback = torch.arange(cells, device=tokens.device).eq(0)
            safe = allowed | (~active_cells[:, :, None] & fallback)
            bias = self.spatial_bias(coords).to(query.dtype).masked_fill(~safe[:, None], float('-inf'))
            # Auto SDPA for the CxC additive bias; within-cell Flash is unchanged.
            context = F.scaled_dot_product_attention(query, key, value, attn_mask=bias,
                dropout_p=self.dropout if self.training else 0.)
        elif self.require_flash and query.is_cuda:
            context = self._flash_cells(query, key, value, allowed)
        else:
            fallback = torch.arange(cells, device=tokens.device).eq(0)
            safe = allowed | (~active_cells[:, :, None] & fallback)
            context = F.scaled_dot_product_attention(query, key, value,
                attn_mask=safe[:, None], dropout_p=self.dropout if self.training else 0.)
        context = context.transpose(1, 2).reshape(batch, cells, dim)
        update = self.out_dropout(self.proj(context)).unsqueeze(2).expand_as(tokens)
        active = cyto & active_cells.unsqueeze(-1)
        return update.masked_fill(~active.unsqueeze(-1), 0.), active


class CompartmentBlock(nn.Module):
    def __init__(self, dim, gene_heads, cell_heads, dropout, ffn_mult,
                 cell_attention_mode="cytoplasm_cross", nuclear_tokens=256,
                 cytoplasmic_tokens=256, cell_token_dim=16, require_flash_attention=False,
                 cell_spatial_bias=False, cell_angle_bins=16,
                 cell_distance_edges=(10., 20., 40., 80., 160., 320.), cell_coordinate_scale=1.):
        super().__init__()
        self.gene_norm1 = nn.LayerNorm(dim)
        self.gene_attn = SDPAAttention(dim, gene_heads, dropout)
        self.gene_attn.require_flash = require_flash_attention
        self.gene_norm2 = nn.LayerNorm(dim)
        self.gene_ffn = FeedForward(dim, ffn_mult, dropout)
        self.cell_norm1 = nn.LayerNorm(dim)
        if cell_attention_mode == "cytoplasm_stack_flat":
            self.cell_attn = StackCytoplasmicCellAttention(
                dim, cell_heads, dropout, nuclear_tokens, cytoplasmic_tokens, cell_token_dim,
                require_flash_attention,
            )
        elif cell_attention_mode == "cytoplasm_mean":
            self.cell_attn = CytoplasmicMeanCellAttention(dim, cell_heads, dropout, require_flash_attention,
                cell_spatial_bias, cell_angle_bins, cell_distance_edges, cell_coordinate_scale)
        elif cell_attention_mode == "cytoplasm_cross":
            self.cell_attn = CytoplasmicCrossAttention(dim, cell_heads, dropout, require_flash_attention)
        else:
            raise ValueError(f"Unsupported compartment attention mode: {cell_attention_mode}")
        self.cell_norm2 = nn.LayerNorm(dim)
        self.cell_ffn = FeedForward(dim, ffn_mult, dropout)

    def within_cell(self, tokens, valid):
        batch, cells, length, dim = tokens.shape
        mask = valid.unsqueeze(-1)
        tokens = tokens.masked_fill(~mask, 0.0)
        normalized = self.gene_norm1(tokens).reshape(batch * cells, length, dim)
        # Remove only padding inside the kernel. This keeps exact standard
        # self-attention while permitting Flash SDPA without a large dense mask.
        flat_valid = valid.flatten(0, 1)
        counts = flat_valid.sum(-1)
        flat_update = torch.zeros_like(normalized).flatten(0, 1)
        scatter_rows, attended_groups = [], []
        for count in counts.unique().tolist():
            if count == 0:
                continue
            rows = counts.eq(count).nonzero(as_tuple=True)[0]
            columns = flat_valid[rows].nonzero(as_tuple=True)[1].reshape(len(rows), count)
            attended = self.gene_attn(normalized[rows[:, None], columns])
            scatter_rows.append((rows[:, None] * length + columns).flatten())
            attended_groups.append(attended.flatten(0, 1).to(flat_update.dtype))
        if scatter_rows:
            flat_update = flat_update.index_copy(0, torch.cat(scatter_rows), torch.cat(attended_groups))
        update = flat_update.reshape_as(normalized)
        update = update + sum(parameter.sum() * 0.0 for parameter in self.gene_attn.parameters())
        tokens = (tokens + update.reshape_as(tokens)).masked_fill(~mask, 0.0)
        return (tokens + self.gene_ffn(self.gene_norm2(tokens))).masked_fill(~mask, 0.0)

    def across_cells(self, tokens, compartments, valid, nodes, coords=None):
        kwargs = {'coords': coords} if isinstance(self.cell_attn, CytoplasmicMeanCellAttention) else {}
        update, active = self.cell_attn(self.cell_norm1(tokens), compartments, valid, nodes, **kwargs)
        output = tokens + update
        ffn = self.cell_ffn(self.cell_norm2(output)).masked_fill(~active.unsqueeze(-1), 0.0)
        return (output + ffn).masked_fill(~valid.unsqueeze(-1), 0.0)

    def forward(self, tokens, compartments, valid, nodes, coords=None):
        # All receiving cells use the same post-within-cell snapshot.
        return self.across_cells(self.within_cell(tokens, valid), compartments, valid, nodes, coords)


class CompartmentSSGTModel(nn.Module):
    def __init__(self, *, num_genes, num_value_bins, hidden_dim, num_layers,
                 gene_heads, cell_heads, dropout, ffn_mult, gene_padding_idx,
                 gene_embedding_init=None, gradient_checkpointing=False,
                 cell_attention_mode="cytoplasm_cross", nuclear_tokens=256,
                 cytoplasmic_tokens=256, cell_token_dim=16, require_flash_attention=False,
                 cell_spatial_bias=False, cell_angle_bins=16,
                 cell_distance_edges=(10., 20., 40., 80., 160., 320.), cell_coordinate_scale=1.):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.cell_attention_mode = cell_attention_mode
        if cell_spatial_bias and cell_attention_mode != 'cytoplasm_mean':
            raise ValueError('Spatial bias is implemented for cytoplasm_mean only')
        self.nuclear_tokens = nuclear_tokens
        self.cytoplasmic_tokens = cytoplasmic_tokens
        self.pad_value_id = num_value_bins + 1
        self.gene_embedding = GeneEmbeddingStem(
            num_genes, hidden_dim, gene_embedding_init, gene_padding_idx,
        )
        self.value_embedding = nn.Embedding(num_value_bins + 2, hidden_dim, padding_idx=self.pad_value_id)
        self.compartment_embedding = nn.Embedding(3, hidden_dim, padding_idx=2)
        self.role_embedding = nn.Embedding(2, hidden_dim)
        self.blocks = nn.ModuleList([
            CompartmentBlock(hidden_dim, gene_heads, cell_heads, dropout, ffn_mult,
                             cell_attention_mode, nuclear_tokens, cytoplasmic_tokens, cell_token_dim,
                             require_flash_attention, cell_spatial_bias, cell_angle_bins,
                             cell_distance_edges, cell_coordinate_scale)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.mask_head = nn.Linear(hidden_dim, num_value_bins)

    def forward(self, *, gene_ids, input_bins, cell_roles, compartment_ids,
                node_indices, token_valid_mask=None, return_cell_attention_weights=False,
                return_embeddings=False, coords=None):
        if return_cell_attention_weights:
            raise ValueError("Returning cross-cell attention weights is not enabled for this model")
        valid = input_bins.ne(self.pad_value_id) if token_valid_mask is None else token_valid_mask.bool()
        if compartment_ids.shape != gene_ids.shape or node_indices.shape != gene_ids.shape[:2]:
            raise ValueError("Compartment IDs or original cell IDs do not match token shape")
        if self.cell_attention_mode == "cytoplasm_stack_flat":
            if gene_ids.shape[2] != self.nuclear_tokens + self.cytoplasmic_tokens:
                raise ValueError("STACK mode requires fixed nuclear and cytoplasmic blocks")
            split = self.nuclear_tokens
            wrong_nuclear = valid[:, :, :split] & compartment_ids[:, :, :split].ne(0)
            wrong_cyto = valid[:, :, split:] & compartment_ids[:, :, split:].ne(1)
            if wrong_nuclear.any() or wrong_cyto.any():
                raise ValueError("Valid tokens must follow nuclear-block then cytoplasmic-block order")
        tokens = (self.gene_embedding(gene_ids) + self.value_embedding(input_bins)
                  + self.compartment_embedding(compartment_ids)
                  + self.role_embedding(cell_roles).unsqueeze(2))
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                # Pass the module itself: no late-bound loop-variable closure.
                tokens = checkpoint(block, tokens, compartment_ids, valid, node_indices, coords, use_reentrant=False)
            else:
                tokens = block(tokens, compartment_ids, valid, node_indices, coords)
        hidden = self.final_norm(tokens).masked_fill(~valid.unsqueeze(-1), 0.0)
        if return_embeddings:
            pooling_mask = valid & compartment_ids.eq(1) if self.cell_attention_mode == "cytoplasm_mean" else valid
            return {"cell_embeddings": hidden.float().masked_fill(~pooling_mask.unsqueeze(-1), 0.).sum(2)
                    / pooling_mask.sum(2).clamp_min(1).unsqueeze(-1)}
        return {"mask_logits": self.mask_head(hidden), "cell_attention_weights": None,
                "cell_attention": None}
