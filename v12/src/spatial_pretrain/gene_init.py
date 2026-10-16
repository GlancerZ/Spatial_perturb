from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from spatial_pretrain.config import Config


def resolve_slice_specs(config: Config) -> list[tuple[str, str, str]]:
    multi_h5ad_paths = list(config.data.h5ad_paths)
    multi_cache_dirs = list(config.data.cache_dirs)
    if multi_h5ad_paths or multi_cache_dirs:
        if len(multi_h5ad_paths) != len(multi_cache_dirs):
            raise ValueError("data.h5ad_paths and data.cache_dirs must have the same length")
        if not multi_h5ad_paths:
            raise ValueError("Multi-slice mode requires at least one h5ad/cache pair")
        return [
            (h5ad_path, cache_dir, Path(cache_dir).name)
            for h5ad_path, cache_dir in zip(multi_h5ad_paths, multi_cache_dirs)
        ]
    return [(config.data.h5ad_path, config.data.cache_dir, Path(config.data.cache_dir).name)]


def build_global_gene_vocabulary(gene_name_lists: Iterable[Iterable[str]]) -> list[str]:
    gene_to_idx: dict[str, int] = {}
    ordered: list[str] = []
    for gene_names in gene_name_lists:
        for gene_name in gene_names:
            if gene_name in gene_to_idx:
                continue
            gene_to_idx[gene_name] = len(ordered)
            ordered.append(gene_name)
    return ordered


def build_global_gene_vocabulary_from_cache_dirs(cache_dirs: Iterable[str | Path]) -> list[str]:
    return build_global_gene_vocabulary(
        Path(cache_dir).joinpath("gene_names.txt").read_text(encoding="utf-8").splitlines()
        for cache_dir in cache_dirs
    )


def build_global_gene_maps(caches) -> tuple[int, list[torch.Tensor]]:
    global_gene_names = build_global_gene_vocabulary(cache.gene_names for cache in caches)
    gene_to_idx = {gene_name: idx for idx, gene_name in enumerate(global_gene_names)}
    mappings: list[torch.Tensor] = []
    for cache in caches:
        mapping = torch.empty((cache.num_genes,), dtype=torch.long)
        for gene_idx, gene_name in enumerate(cache.gene_names):
            mapping[gene_idx] = gene_to_idx[gene_name]
        mappings.append(mapping)
    return len(global_gene_names), mappings


def load_gene_init_artifact(
    path: str | Path,
    *,
    num_genes: int,
    hidden_dim: int,
    gene_names: list[str] | None = None,
) -> torch.Tensor:
    raw = torch.load(Path(path), map_location="cpu")
    full_embedding_dim = None
    if isinstance(raw, dict):
        if "embeddings" not in raw:
            raise KeyError(f"Gene init artifact {path} is missing the 'embeddings' key")
        full_embedding_dim = raw.get("full_embedding_dim")
        if gene_names is not None:
            artifact_names = raw.get("gene_names")
            if artifact_names is None:
                raise ValueError("Vocabulary alignment requires gene_names in the GenePT artifact")
            if len(artifact_names) != len(set(artifact_names)):
                raise ValueError("GenePT artifact contains duplicate gene names")
            lookup = {name: i for i, name in enumerate(artifact_names)}
            missing = [name for name in gene_names if name not in lookup]
            if missing:
                raise ValueError(f"GenePT is missing {len(missing)} genes: {missing[:10]}")
            raw["embeddings"] = raw["embeddings"][[lookup[name] for name in gene_names]]
        raw = raw["embeddings"]
    elif gene_names is not None:
        raise ValueError("Vocabulary alignment requires a named GenePT artifact")
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Expected a tensor or dict-backed tensor at {path}, got {type(raw)}")
    if raw.ndim != 2:
        raise ValueError(f"Gene init artifact at {path} must be rank-2, got shape {tuple(raw.shape)}")
    if raw.shape[0] != num_genes:
        raise ValueError(f"Gene init rows {raw.shape[0]} != num_genes {num_genes}")
    if raw.shape[1] != hidden_dim and full_embedding_dim is not None and raw.shape[1] < int(full_embedding_dim):
        raise ValueError(
            f"Gene init cols {raw.shape[1]} != hidden_dim {hidden_dim}, and the artifact stores a truncated "
            f"subvector from a full Qwen embedding dim {full_embedding_dim}. Rebuild it with full Qwen embeddings "
            f"before using the MLP projector path."
        )
    return raw.to(dtype=torch.float32, copy=False)
