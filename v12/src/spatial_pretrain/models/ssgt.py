from __future__ import annotations

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    from torch.nn.attention.flex_attention import flex_attention
except ImportError:
    flex_attention = None

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float) -> None:
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeneEmbeddingStem(nn.Module):
    def __init__(
        self,
        num_genes: int,
        hidden_dim: int,
        gene_embedding_init: torch.Tensor | None = None,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.padding_idx = padding_idx
        if gene_embedding_init is None or gene_embedding_init.shape[1] == hidden_dim:
            self.embedding = nn.Embedding(num_genes, hidden_dim, padding_idx=padding_idx)
            if gene_embedding_init is not None:
                if gene_embedding_init.shape != self.embedding.weight.shape:
                    raise ValueError(
                        f"gene_embedding_init shape {tuple(gene_embedding_init.shape)} does not match "
                        f"embedding shape {tuple(self.embedding.weight.shape)}"
                    )
                with torch.no_grad():
                    self.embedding.weight.copy_(gene_embedding_init)
                    if padding_idx is not None:
                        self.embedding.weight[padding_idx].zero_()
            self.register_buffer("raw_gene_features", None, persistent=False)
            self.projector = None
            return

        if gene_embedding_init.shape[0] != num_genes:
            raise ValueError(
                f"gene_embedding_init rows {gene_embedding_init.shape[0]} do not match num_genes {num_genes}"
            )
        self.embedding = None
        self.register_buffer("raw_gene_features", gene_embedding_init.contiguous(), persistent=True)
        self.projector = nn.Sequential(
            nn.Linear(int(gene_embedding_init.shape[1]), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, gene_ids: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            output = self.embedding(gene_ids)
            if self.padding_idx is not None:
                output = output.masked_fill(gene_ids.eq(self.padding_idx).unsqueeze(-1), 0.0)
            return output
        if self.raw_gene_features is None or self.projector is None:
            raise RuntimeError("GeneEmbeddingStem is missing both embedding weights and raw projector inputs")
        output = self.projector(self.raw_gene_features[gene_ids])
        if self.padding_idx is not None:
            output = output.masked_fill(gene_ids.eq(self.padding_idx).unsqueeze(-1), 0.0)
        return output


def _prefer_flash_sdpa():
    if sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH])


def _can_use_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if flash_attn_func is None:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.shape[-1] > 256:
        return False
    return True


def _can_use_flex_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    if flex_attention is None:
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    return True


class SDPAAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        key_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        batch_size, query_len, _ = query.shape
        q = self.q_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        key_len = key.shape[1]
        k = self.k_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim)
        value_len = value.shape[1]
        v = self.v_proj(value).view(batch_size, value_len, self.num_heads, self.head_dim)

        if key_valid_mask is not None:
            key_valid_mask = key_valid_mask.to(device=query.device, dtype=torch.bool)
            if key_valid_mask.shape != (batch_size, key_len):
                raise ValueError(
                    f"key_valid_mask shape {tuple(key_valid_mask.shape)} does not match "
                    f"(batch_size, key_len)=({batch_size}, {key_len})"
                )
            if not bool(key_valid_mask.all()):
                safe_key_valid_mask = key_valid_mask
                empty_keys = ~safe_key_valid_mask.any(dim=1)
                if bool(empty_keys.any()):
                    safe_key_valid_mask = safe_key_valid_mask.clone()
                    safe_key_valid_mask[empty_keys, 0] = True
                attn_mask = torch.zeros(
                    (batch_size, 1, 1, key_len),
                    dtype=q.dtype,
                    device=q.device,
                )
                attn_mask = attn_mask.masked_fill(
                    ~safe_key_valid_mask[:, None, None, :],
                    torch.finfo(q.dtype).min,
                )
            else:
                attn_mask = None
        else:
            attn_mask = None

        if getattr(self, "require_flash", False) and q.is_cuda:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0,
                )
            out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.dim)
        elif key_valid_mask is None and _can_use_flash_attn(q, k, v):
            out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False,
            )
            out = out.contiguous().view(batch_size, query_len, self.dim)
        else:
            with _prefer_flash_sdpa():
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            out = out.transpose(1, 2).contiguous().view(batch_size, query_len, self.dim)
        return self.out_dropout(self.out_proj(out))


class FlattenedCellAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        pair_dim: int | None = None,
        mode: str = "identity_flat",
        cell_token_count: int | None = None,
        mla_latent_dim: int = 128,
        mla_qk_head_dim: int = 64,
        mla_v_head_dim: int = 64,
        mla_output_rank: int = 128,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.mode = mode.strip().lower()
        self.cell_token_count = cell_token_count
        self.mla_latent_dim = mla_latent_dim
        self.mla_qk_head_dim = mla_qk_head_dim
        self.mla_v_head_dim = mla_v_head_dim
        self.mla_output_rank = mla_output_rank
        if self.mode not in {"identity_flat", "learned_qkv", "cell_token_qkv", "mla_flat", "mla_q_kv_flat"}:
            raise ValueError(f"Unsupported cell attention mode {mode!r}")

        if self.mode == "learned_qkv":
            # Keep the flattened cross-cell attention pattern, but learn tokenwise
            # Q/K/V projections before flattening so parameter count stays tractable.
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.out_dropout = nn.Dropout(dropout)
            self.cell_pool_query = None
            self.cell_pool_attn = None
        elif self.mode == "cell_token_qkv":
            # Compress each cell's token table into one hidden-dim cell token first,
            # then run standard self-attention across those cell tokens.
            self.cell_pool_query = nn.Parameter(torch.randn(1, 1, dim) * (dim ** -0.5))
            self.cell_pool_attn = SDPAAttention(dim, num_heads, dropout)
            self.q_proj = nn.Linear(dim, dim)
            self.k_proj = nn.Linear(dim, dim)
            self.v_proj = nn.Linear(dim, dim)
            self.out_proj = nn.Linear(dim, dim)
            self.out_dropout = nn.Dropout(dropout)
            self.mla_latent_proj = None
            self.mla_q_up_proj = None
            self.mla_k_up_proj = None
            self.mla_v_up_proj = None
            self.mla_out_down_proj = None
            self.mla_out_up_proj = None
        elif self.mode == "mla_flat":
            if cell_token_count is None or cell_token_count <= 0:
                raise ValueError("mla_flat mode requires a positive cell_token_count")
            flat_dim = int(cell_token_count) * dim
            self.mla_latent_proj = nn.Linear(flat_dim, mla_latent_dim, bias=False)
            self.mla_q_up_proj = nn.Linear(mla_latent_dim, num_heads * mla_qk_head_dim, bias=False)
            self.mla_k_up_proj = nn.Linear(mla_latent_dim, num_heads * mla_qk_head_dim, bias=False)
            self.mla_v_up_proj = nn.Linear(mla_latent_dim, num_heads * mla_v_head_dim, bias=False)
            self.mla_out_down_proj = nn.Linear(num_heads * mla_v_head_dim, mla_output_rank, bias=False)
            self.mla_out_up_proj = nn.Linear(mla_output_rank, flat_dim, bias=False)
            self.out_dropout = nn.Dropout(dropout)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.cell_pool_query = None
            self.cell_pool_attn = None
            self.mla_q_latent_proj = None
            self.mla_kv_latent_proj = None
        elif self.mode == "mla_q_kv_flat":
            if cell_token_count is None or cell_token_count <= 0:
                raise ValueError("mla_q_kv_flat mode requires a positive cell_token_count")
            flat_dim = int(cell_token_count) * dim
            self.mla_q_latent_proj = nn.Linear(flat_dim, mla_latent_dim, bias=False)
            self.mla_kv_latent_proj = nn.Linear(flat_dim, mla_latent_dim, bias=False)
            self.mla_q_up_proj = nn.Linear(mla_latent_dim, num_heads * mla_qk_head_dim, bias=False)
            self.mla_k_up_proj = nn.Linear(mla_latent_dim, num_heads * mla_qk_head_dim, bias=False)
            self.mla_v_up_proj = nn.Linear(mla_latent_dim, num_heads * mla_v_head_dim, bias=False)
            self.mla_out_down_proj = nn.Linear(num_heads * mla_v_head_dim, mla_output_rank, bias=False)
            self.mla_out_up_proj = nn.Linear(mla_output_rank, flat_dim, bias=False)
            self.out_dropout = nn.Dropout(dropout)
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.cell_pool_query = None
            self.cell_pool_attn = None
            self.mla_latent_proj = None
        else:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.out_proj = None
            self.out_dropout = None
            self.cell_pool_query = None
            self.cell_pool_attn = None
            self.mla_latent_proj = None
            self.mla_q_latent_proj = None
            self.mla_kv_latent_proj = None
            self.mla_q_up_proj = None
            self.mla_k_up_proj = None
            self.mla_v_up_proj = None
            self.mla_out_down_proj = None
            self.mla_out_up_proj = None

    def _forward_mla_flat_attention(
        self,
        cell_tokens: torch.Tensor,
        lr_bias: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_cells, _ = cell_tokens.shape
        if self.cell_token_count is None:
            raise RuntimeError("mla_flat mode requires cell_token_count")
        if seq_len != self.cell_token_count:
            raise ValueError(
                f"mla_flat mode expected seq_len={self.cell_token_count}, got seq_len={seq_len}"
            )
        if any(
            layer is None
            for layer in (
                self.mla_latent_proj,
                self.mla_q_up_proj,
                self.mla_k_up_proj,
                self.mla_v_up_proj,
                self.mla_out_down_proj,
                self.mla_out_up_proj,
                self.out_dropout,
            )
        ):
            raise RuntimeError("mla_flat mode is missing one or more low-rank projection layers")

        flat_dim = seq_len * self.dim
        scale = self.mla_qk_head_dim ** -0.5
        flat_cells = cell_tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, num_cells, flat_dim)
        latent = self.mla_latent_proj(flat_cells)
        query = self.mla_q_up_proj(latent).view(batch_size, num_cells, self.num_heads, self.mla_qk_head_dim).transpose(1, 2)
        key = self.mla_k_up_proj(latent).view(batch_size, num_cells, self.num_heads, self.mla_qk_head_dim).transpose(1, 2)
        value = self.mla_v_up_proj(latent).view(batch_size, num_cells, self.num_heads, self.mla_v_head_dim).transpose(1, 2)

        use_flex_bias_path = (
            lr_bias is not None
            and not return_attention_weights
            and _can_use_flex_attention(query, key, value)
        )
        if use_flex_bias_path:
            lr_bias_local = lr_bias.contiguous().to(dtype=query.dtype)

            def score_mod(
                score: torch.Tensor,
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del head
                return score + lr_bias_local[batch, q_idx, kv_idx]

            attention_context = flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                scale=scale,
            )
        else:
            attn_mask = None
            if lr_bias is not None:
                attn_mask = lr_bias[:, None, :, :].to(dtype=query.dtype)
            with _prefer_flash_sdpa():
                attention_context = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )

        attention_context = attention_context.transpose(1, 2).contiguous().view(
            batch_size,
            num_cells,
            self.num_heads * self.mla_v_head_dim,
        )
        attention_context = self.mla_out_down_proj(attention_context)
        attention_output = self.mla_out_up_proj(attention_context)
        attention_output = self.out_dropout(attention_output)
        attention_output = attention_output.view(batch_size, num_cells, seq_len, self.dim).permute(0, 2, 1, 3).contiguous()

        attention_weights = None
        if return_attention_weights:
            attn_logits = torch.matmul(query, key.transpose(-1, -2)) * scale
            if lr_bias is not None:
                attn_logits = attn_logits + lr_bias[:, None, :, :].to(dtype=attn_logits.dtype)
            attention_weights = torch.softmax(attn_logits, dim=-1)
        return attention_output, attention_weights

    def _forward_mla_q_kv_flat_attention(
        self,
        cell_tokens: torch.Tensor,
        lr_bias: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_cells, _ = cell_tokens.shape
        if self.cell_token_count is None:
            raise RuntimeError("mla_q_kv_flat mode requires cell_token_count")
        if seq_len != self.cell_token_count:
            raise ValueError(
                f"mla_q_kv_flat mode expected seq_len={self.cell_token_count}, got seq_len={seq_len}"
            )
        if any(
            layer is None
            for layer in (
                self.mla_q_latent_proj,
                self.mla_kv_latent_proj,
                self.mla_q_up_proj,
                self.mla_k_up_proj,
                self.mla_v_up_proj,
                self.mla_out_down_proj,
                self.mla_out_up_proj,
                self.out_dropout,
            )
        ):
            raise RuntimeError("mla_q_kv_flat mode is missing one or more MLA projection layers")

        flat_dim = seq_len * self.dim
        scale = self.mla_qk_head_dim ** -0.5
        flat_cells = cell_tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, num_cells, flat_dim)

        query_latent = self.mla_q_latent_proj(flat_cells)
        kv_latent = self.mla_kv_latent_proj(flat_cells)
        query = self.mla_q_up_proj(query_latent).view(
            batch_size,
            num_cells,
            self.num_heads,
            self.mla_qk_head_dim,
        ).transpose(1, 2)
        key = self.mla_k_up_proj(kv_latent).view(
            batch_size,
            num_cells,
            self.num_heads,
            self.mla_qk_head_dim,
        ).transpose(1, 2)
        value = self.mla_v_up_proj(kv_latent).view(
            batch_size,
            num_cells,
            self.num_heads,
            self.mla_v_head_dim,
        ).transpose(1, 2)

        use_flex_bias_path = (
            lr_bias is not None
            and not return_attention_weights
            and _can_use_flex_attention(query, key, value)
        )
        if use_flex_bias_path:
            lr_bias_local = lr_bias.contiguous().to(dtype=query.dtype)

            def score_mod(
                score: torch.Tensor,
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del head
                return score + lr_bias_local[batch, q_idx, kv_idx]

            attention_context = flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                scale=scale,
            )
        else:
            attn_mask = None
            if lr_bias is not None:
                attn_mask = lr_bias[:, None, :, :].to(dtype=query.dtype)
            with _prefer_flash_sdpa():
                attention_context = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )

        attention_context = attention_context.transpose(1, 2).contiguous().view(
            batch_size,
            num_cells,
            self.num_heads * self.mla_v_head_dim,
        )
        attention_context = self.mla_out_down_proj(attention_context)
        attention_output = self.mla_out_up_proj(attention_context)
        attention_output = self.out_dropout(attention_output)
        attention_output = attention_output.view(batch_size, num_cells, seq_len, self.dim).permute(0, 2, 1, 3).contiguous()

        attention_weights = None
        if return_attention_weights:
            attn_logits = torch.matmul(query, key.transpose(-1, -2)) * scale
            if lr_bias is not None:
                attn_logits = attn_logits + lr_bias[:, None, :, :].to(dtype=attn_logits.dtype)
            attention_weights = torch.softmax(attn_logits, dim=-1)
        return attention_output, attention_weights

    def _forward_cell_token_attention(
        self,
        cell_tokens: torch.Tensor,
        lr_bias: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, num_cells, _ = cell_tokens.shape
        if self.dim % self.num_heads != 0:
            raise ValueError(f"dim={self.dim} must be divisible by num_heads={self.num_heads}")
        head_dim = self.dim // self.num_heads
        scale = head_dim ** -0.5

        tokens_by_cell = cell_tokens.permute(0, 2, 1, 3).contiguous().view(batch_size * num_cells, seq_len, self.dim)
        if self.cell_pool_query is None or self.cell_pool_attn is None:
            raise RuntimeError("cell_token_qkv mode requires cell pooling modules")
        pool_query = self.cell_pool_query.expand(batch_size * num_cells, -1, -1)
        pool_key_valid_mask = (
            token_valid_mask.reshape(batch_size * num_cells, seq_len)
            if token_valid_mask is not None
            else None
        )
        cell_token_inputs = self.cell_pool_attn(
            pool_query,
            tokens_by_cell,
            tokens_by_cell,
            key_valid_mask=pool_key_valid_mask,
        ).squeeze(1)
        cell_token_inputs = cell_token_inputs.view(batch_size, num_cells, self.dim)

        if any(layer is None for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.out_dropout)):
            raise RuntimeError("cell_token_qkv mode requires Q/K/V projection layers")
        query = self.q_proj(cell_token_inputs).view(batch_size, num_cells, self.num_heads, head_dim).transpose(1, 2)
        key = self.k_proj(cell_token_inputs).view(batch_size, num_cells, self.num_heads, head_dim).transpose(1, 2)
        value = self.v_proj(cell_token_inputs).view(batch_size, num_cells, self.num_heads, head_dim).transpose(1, 2)

        use_flex_bias_path = (
            lr_bias is not None
            and not return_attention_weights
            and _can_use_flex_attention(query, key, value)
        )
        if use_flex_bias_path:
            lr_bias_local = lr_bias.contiguous().to(dtype=query.dtype)

            def score_mod(
                score: torch.Tensor,
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del head
                return score + lr_bias_local[batch, q_idx, kv_idx]

            cell_context = flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                scale=scale,
            )
        else:
            attn_mask = None
            if lr_bias is not None:
                attn_mask = lr_bias[:, None, :, :].to(dtype=query.dtype)
            with _prefer_flash_sdpa():
                cell_context = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )

        cell_context = cell_context.transpose(1, 2).contiguous().view(batch_size, num_cells, self.dim)
        cell_context = self.out_dropout(self.out_proj(cell_context))
        attention_output = cell_context.unsqueeze(1).expand(batch_size, seq_len, num_cells, self.dim)

        attention_weights = None
        if return_attention_weights:
            attn_logits = torch.matmul(query, key.transpose(-1, -2)) * scale
            if lr_bias is not None:
                attn_logits = attn_logits + lr_bias[:, None, :, :].to(dtype=attn_logits.dtype)
            attention_weights = torch.softmax(attn_logits, dim=-1)
        return attention_output, attention_weights

    def forward(
        self,
        cell_tokens: torch.Tensor,
        lr_bias: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.mode == "mla_flat":
            return self._forward_mla_flat_attention(
                cell_tokens,
                lr_bias,
                return_attention_weights=return_attention_weights,
            )
        if self.mode == "mla_q_kv_flat":
            return self._forward_mla_q_kv_flat_attention(
                cell_tokens,
                lr_bias,
                return_attention_weights=return_attention_weights,
            )
        if self.mode == "cell_token_qkv":
            return self._forward_cell_token_attention(
                cell_tokens,
                lr_bias,
                token_valid_mask=token_valid_mask,
                return_attention_weights=return_attention_weights,
            )

        batch_size, seq_len, num_cells, _ = cell_tokens.shape
        flat_dim = seq_len * self.dim
        if flat_dim % self.num_heads != 0:
            raise ValueError(
                f"Flattened cell embedding dim={flat_dim} must be divisible by num_heads={self.num_heads}"
            )
        head_dim = flat_dim // self.num_heads
        scale = head_dim ** -0.5

        if self.mode == "learned_qkv":
            query_tokens = self.q_proj(cell_tokens)
            key_tokens = self.k_proj(cell_tokens)
            value_tokens = self.v_proj(cell_tokens)
        else:
            query_tokens = cell_tokens
            key_tokens = cell_tokens
            value_tokens = cell_tokens

        # Collapse all gene-token features inside each cell into a single flattened
        # cell embedding, then attend across cells before reshaping back.
        query_cells = query_tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, num_cells, flat_dim)
        key_cells = key_tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, num_cells, flat_dim)
        value_cells = value_tokens.permute(0, 2, 1, 3).contiguous().view(batch_size, num_cells, flat_dim)
        query = query_cells.view(batch_size, num_cells, self.num_heads, head_dim).transpose(1, 2)
        key = key_cells.view(batch_size, num_cells, self.num_heads, head_dim).transpose(1, 2)
        value = value_cells.view(batch_size, num_cells, self.num_heads, head_dim).transpose(1, 2)

        use_flex_bias_path = (
            lr_bias is not None
            and not return_attention_weights
            and _can_use_flex_attention(query, key, value)
        )
        if use_flex_bias_path:
            lr_bias_local = lr_bias.contiguous().to(dtype=query.dtype)

            def score_mod(
                score: torch.Tensor,
                batch: torch.Tensor,
                head: torch.Tensor,
                q_idx: torch.Tensor,
                kv_idx: torch.Tensor,
            ) -> torch.Tensor:
                del head
                return score + lr_bias_local[batch, q_idx, kv_idx]

            # flex_attention supports score-space bias injection efficiently, but
            # unlike SDPA it does not currently expose a dropout_p argument.
            attention_output = flex_attention(
                query,
                key,
                value,
                score_mod=score_mod,
                scale=scale,
            )
        else:
            attn_mask = None
            if lr_bias is not None:
                attn_mask = lr_bias[:, None, :, :].to(dtype=query.dtype)
            with _prefer_flash_sdpa():
                attention_output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, num_cells, seq_len, self.dim)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        if self.out_proj is not None and self.out_dropout is not None:
            attention_output = self.out_dropout(self.out_proj(attention_output))

        attention_weights = None
        if return_attention_weights:
            attn_logits = torch.matmul(query, key.transpose(-1, -2)) * scale
            if lr_bias is not None:
                attn_logits = attn_logits + lr_bias[:, None, :, :].to(dtype=attn_logits.dtype)
            attention_weights = torch.softmax(attn_logits, dim=-1)
        return attention_output, attention_weights


BiasAwareCellAttention = FlattenedCellAttention


class HierarchicalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        gene_heads: int,
        cell_heads: int,
        dropout: float,
        ffn_mult: int,
        pair_dim: int,
        cell_attention_mode: str = "identity_flat",
        cell_token_count: int | None = None,
        cell_mla_latent_dim: int = 128,
        cell_mla_qk_head_dim: int = 64,
        cell_mla_v_head_dim: int = 64,
        cell_mla_output_rank: int = 128,
    ) -> None:
        super().__init__()
        self.gene_norm1 = nn.LayerNorm(dim)
        self.gene_attn = SDPAAttention(dim, gene_heads, dropout)
        self.gene_norm2 = nn.LayerNorm(dim)
        self.gene_ffn = FeedForward(dim, ffn_mult, dropout)

        self.cell_norm1 = nn.LayerNorm(dim)
        self.cell_attn = FlattenedCellAttention(
            dim,
            cell_heads,
            dropout,
            pair_dim,
            mode=cell_attention_mode,
            cell_token_count=cell_token_count,
            mla_latent_dim=cell_mla_latent_dim,
            mla_qk_head_dim=cell_mla_qk_head_dim,
            mla_v_head_dim=cell_mla_v_head_dim,
            mla_output_rank=cell_mla_output_rank,
        )
        self.cell_norm2 = nn.LayerNorm(dim)
        self.cell_ffn = FeedForward(dim, ffn_mult, dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        lr_bias: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        *,
        return_cell_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, num_cells, seq_len, dim = tokens.shape
        valid = token_valid_mask.unsqueeze(-1) if token_valid_mask is not None else None
        if valid is not None:
            tokens = tokens.masked_fill(~valid, 0.0)

        gene_input = self.gene_norm1(tokens).view(batch_size * num_cells, seq_len, dim)
        gene_key_valid_mask = (
            token_valid_mask.reshape(batch_size * num_cells, seq_len)
            if token_valid_mask is not None
            else None
        )
        gene_out = self.gene_attn(gene_input, key_valid_mask=gene_key_valid_mask)
        gene_out = gene_out.view(batch_size, num_cells, seq_len, dim)
        if valid is not None:
            gene_out = gene_out.masked_fill(~valid, 0.0)
        tokens = tokens + gene_out
        if valid is not None:
            tokens = tokens.masked_fill(~valid, 0.0)
        tokens = tokens + self.gene_ffn(self.gene_norm2(tokens))
        if valid is not None:
            tokens = tokens.masked_fill(~valid, 0.0)

        # Flatten each cell's full token table into one large cell embedding, apply
        # attention across cells, then reshape back to token space.
        cell_input = (
            self.cell_norm1(tokens)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        cell_out, cell_attention_weights = self.cell_attn(
            cell_input,
            lr_bias,
            token_valid_mask=token_valid_mask,
            return_attention_weights=return_cell_attention_weights,
        )
        cell_out = (
            cell_out.permute(0, 2, 1, 3)
            .contiguous()
        )
        if valid is not None:
            cell_out = cell_out.masked_fill(~valid, 0.0)
        tokens = tokens + cell_out
        if valid is not None:
            tokens = tokens.masked_fill(~valid, 0.0)
        tokens = tokens + self.cell_ffn(self.cell_norm2(tokens))
        if valid is not None:
            tokens = tokens.masked_fill(~valid, 0.0)
        return tokens, cell_attention_weights


class SSGTModel(nn.Module):
    def __init__(
        self,
        *,
        num_genes: int,
        num_value_bins: int,
        hidden_dim: int,
        num_layers: int,
        gene_heads: int,
        cell_heads: int,
        dropout: float,
        ffn_mult: int,
        use_center_head: bool = True,
        gene_embedding_init: torch.Tensor | None = None,
        cell_attention_mode: str = "identity_flat",
        cell_token_count: int | None = None,
        cell_mla_latent_dim: int = 128,
        cell_mla_qk_head_dim: int = 64,
        cell_mla_v_head_dim: int = 64,
        cell_mla_output_rank: int = 128,
        use_padding_tokens: bool = False,
        gene_padding_idx: int | None = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.num_value_bins = num_value_bins
        self.mask_token_id = num_value_bins
        self.use_padding_tokens = use_padding_tokens
        self.pad_value_id = num_value_bins + 1 if use_padding_tokens else None
        self.gene_padding_idx = gene_padding_idx if use_padding_tokens else None
        self.use_center_head = use_center_head
        self.cell_attention_mode = cell_attention_mode.strip().lower()
        self.gradient_checkpointing = gradient_checkpointing

        self.gene_embedding = GeneEmbeddingStem(
            num_genes=num_genes,
            hidden_dim=hidden_dim,
            gene_embedding_init=gene_embedding_init,
            padding_idx=self.gene_padding_idx,
        )

        self.value_embedding = nn.Embedding(
            num_value_bins + (2 if use_padding_tokens else 1),
            hidden_dim,
            padding_idx=self.pad_value_id,
        )
        self.role_embedding = nn.Embedding(2, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                HierarchicalBlock(
                    dim=hidden_dim,
                    gene_heads=gene_heads,
                    cell_heads=cell_heads,
                    dropout=dropout,
                    ffn_mult=ffn_mult,
                    pair_dim=4,
                    cell_attention_mode=self.cell_attention_mode,
                    cell_token_count=cell_token_count,
                    cell_mla_latent_dim=cell_mla_latent_dim,
                    cell_mla_qk_head_dim=cell_mla_qk_head_dim,
                    cell_mla_v_head_dim=cell_mla_v_head_dim,
                    cell_mla_output_rank=cell_mla_output_rank,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.mask_head = nn.Linear(hidden_dim, num_value_bins)
        if self.use_center_head:
            self.center_cross_attn = SDPAAttention(hidden_dim, gene_heads, dropout)
            self.center_norm = nn.LayerNorm(hidden_dim)
            self.center_head = nn.Linear(hidden_dim, num_value_bins)

    def forward(
        self,
        *,
        gene_ids: torch.Tensor,
        input_bins: torch.Tensor,
        cell_roles: torch.Tensor,
        lr_bias: torch.Tensor | None = None,
        token_valid_mask: torch.Tensor | None = None,
        return_cell_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        if token_valid_mask is None and self.use_padding_tokens:
            if self.pad_value_id is None:
                raise RuntimeError("Padding tokens are enabled but pad_value_id is unset")
            token_valid_mask = input_bins.ne(self.pad_value_id)
        if token_valid_mask is not None:
            token_valid_mask = token_valid_mask.to(device=input_bins.device, dtype=torch.bool)

        gene_emb = self.gene_embedding(gene_ids)
        value_emb = self.value_embedding(input_bins)
        role_emb = self.role_embedding(cell_roles).unsqueeze(2)
        tokens = gene_emb + value_emb + role_emb
        if token_valid_mask is not None:
            tokens = tokens.masked_fill(~token_valid_mask.unsqueeze(-1), 0.0)

        last_cell_attention_weights = None
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and not return_cell_attention_weights:
                def _block_forward(hidden_states: torch.Tensor, block=block) -> torch.Tensor:
                    block_out, _ = block(
                        hidden_states,
                        lr_bias,
                        token_valid_mask=token_valid_mask,
                        return_cell_attention_weights=False,
                    )
                    return block_out

                tokens = checkpoint(_block_forward, tokens, use_reentrant=False)
                last_cell_attention_weights = None
            else:
                tokens, last_cell_attention_weights = block(
                    tokens,
                    lr_bias,
                    token_valid_mask=token_valid_mask,
                    return_cell_attention_weights=return_cell_attention_weights,
                )

        tokens = self.final_norm(tokens)
        if token_valid_mask is not None:
            tokens = tokens.masked_fill(~token_valid_mask.unsqueeze(-1), 0.0)
        gene_hidden = tokens

        mask_logits = self.mask_head(gene_hidden)

        outputs = {
            "mask_logits": mask_logits,
            "cell_attention_weights": last_cell_attention_weights,
            # Backward-compatible alias for older loss/analysis code.
            "cell_attention": last_cell_attention_weights,
        }
        if self.use_center_head:
            center_queries = gene_hidden[:, 0, :, :]
            neighbor_memory = gene_hidden[:, 1:, :, :].mean(dim=2)
            center_context = self.center_cross_attn(center_queries, neighbor_memory, neighbor_memory)
            center_hidden = self.center_norm(center_queries + center_context)
            outputs["center_logits"] = self.center_head(center_hidden)
        return outputs
