from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from numpy.lib.format import open_memmap
from sklearn.neighbors import NearestNeighbors

from spatial_pretrain.data.lr import load_lr_pairs, map_lr_pairs_to_genes

CACHE_METADATA_VERSION = 2
REQUIRED_CACHE_FILES = (
    "expr_log1p_float16.npy",
    "coords.npy",
    "knn_indices.npy",
    "knn_distances.npy",
    "bin_edges.npy",
    "gene_names.txt",
    "lr_pairs_idx.npy",
    "receptor_gene_ids.npy",
)


def _to_dense_float32(matrix) -> np.ndarray:
    if sp.issparse(matrix):
        return matrix.toarray().astype(np.float32, copy=False)
    return np.asarray(matrix, dtype=np.float32)


def _resolve_recorded_path(path: str | Path, cache_dir: Path) -> list[Path]:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return [raw]

    candidates = [Path.cwd() / raw]
    try:
        resolved_cache = cache_dir.resolve()
    except OSError:
        resolved_cache = cache_dir
    for parent in resolved_cache.parents:
        candidates.append(parent / raw)
    return candidates


def _paths_match(recorded: str | Path | None, expected: Path, cache_dir: Path) -> bool:
    if not recorded:
        return False
    expected = expected.expanduser()
    for candidate in _resolve_recorded_path(recorded, cache_dir):
        if str(candidate) == str(expected):
            return True
        try:
            if candidate.exists() and expected.exists() and candidate.samefile(expected):
                return True
        except OSError:
            continue
    return False


def _lr_db_metadata(path: str | Path) -> dict[str, int | str]:
    if not path:
        return {"lr_db_path": "", "lr_db_size": 0, "lr_db_mtime_ns": 0}
    lr_path = Path(path).expanduser()
    stat = lr_path.stat()
    return {
        "lr_db_path": str(lr_path.resolve()),
        "lr_db_size": int(stat.st_size),
        "lr_db_mtime_ns": int(stat.st_mtime_ns),
    }


def _validate_existing_cache(
    *,
    cache_dir: Path,
    h5ad_path: Path,
    lr_db_path: str | Path,
    knn_k: int,
    num_bins: int,
    max_cells: int,
) -> None:
    missing = [name for name in REQUIRED_CACHE_FILES if not (cache_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Cache {cache_dir} is incomplete; missing files: {missing}")

    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if not _paths_match(metadata.get("h5ad_path"), h5ad_path, cache_dir):
        mismatches.append(
            f"h5ad_path metadata={metadata.get('h5ad_path')!r} expected={str(h5ad_path)!r}"
        )
    for key, expected in (("knn_k", int(knn_k)), ("num_bins", int(num_bins))):
        if int(metadata.get(key, -1)) != expected:
            mismatches.append(f"{key} metadata={metadata.get(key)!r} expected={expected!r}")
    if "max_cells" in metadata and int(metadata["max_cells"]) != int(max_cells):
        mismatches.append(f"max_cells metadata={metadata['max_cells']!r} expected={int(max_cells)!r}")
    if "lr_db_path" in metadata:
        lr_meta = _lr_db_metadata(lr_db_path)
        if (metadata.get("lr_db_path") or lr_meta["lr_db_path"]) and not _paths_match(metadata.get("lr_db_path"), Path(lr_meta["lr_db_path"]), cache_dir):
            mismatches.append(
                f"lr_db_path metadata={metadata.get('lr_db_path')!r} expected={lr_meta['lr_db_path']!r}"
            )
        for key in ("lr_db_size", "lr_db_mtime_ns"):
            if int(metadata.get(key, -1)) != int(lr_meta[key]):
                mismatches.append(f"{key} metadata={metadata.get(key)!r} expected={lr_meta[key]!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(f"Cache {cache_dir} does not match the requested dataset/config: {joined}")


@dataclass
class SliceCache:
    cache_dir: Path
    expr: np.memmap
    coords: np.ndarray
    knn_indices: np.ndarray
    knn_distances: np.ndarray
    bin_edges: np.ndarray
    gene_names: list[str]
    lr_pairs_idx: np.ndarray
    receptor_gene_ids: np.ndarray
    num_cells: int
    num_genes: int
    nuclear_expr: np.ndarray | None = None
    cytoplasmic_expr: np.ndarray | None = None
    bin_offset: int = 0


def build_slice_cache(
    *,
    h5ad_path: str | Path,
    cache_dir: str | Path,
    lr_db_path: str | Path,
    knn_k: int,
    chunk_size: int,
    num_bins: int,
    max_cells: int = 0,
) -> Path:
    h5ad_path = Path(h5ad_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    meta_path = cache_dir / "metadata.json"
    if meta_path.exists():
        _validate_existing_cache(
            cache_dir=cache_dir,
            h5ad_path=h5ad_path,
            lr_db_path=lr_db_path,
            knn_k=knn_k,
            num_bins=num_bins,
            max_cells=max_cells,
        )
        return cache_dir

    adata = ad.read_h5ad(h5ad_path, backed="r")
    total_cells, num_genes = adata.shape
    num_cells = min(total_cells, max_cells) if max_cells and max_cells > 0 else total_cells

    coords = np.asarray(adata.obsm["spatial"][:num_cells], dtype=np.float32)
    np.save(cache_dir / "coords.npy", coords)

    expr_path = cache_dir / "expr_log1p_float16.npy"
    expr_memmap = open_memmap(expr_path, mode="w+", dtype=np.float16, shape=(num_cells, num_genes))

    collected_nonzero: list[np.ndarray] = []
    target_nonzero_samples = 2_000_000
    rng = np.random.default_rng(42)

    for start in range(0, num_cells, chunk_size):
        stop = min(start + chunk_size, num_cells)
        chunk = adata[start:stop].X
        dense = np.log1p(_to_dense_float32(chunk))
        expr_memmap[start:stop] = dense.astype(np.float16)

        nonzero = dense[dense > 0]
        if nonzero.size > 0:
            if nonzero.size > 50_000:
                picked = rng.choice(nonzero, size=50_000, replace=False)
            else:
                picked = nonzero
            collected_nonzero.append(np.asarray(picked, dtype=np.float32))

        print(f"[cache] wrote expression rows {start}:{stop}", flush=True)

    sampled_values = np.concatenate(collected_nonzero) if collected_nonzero else np.array([0.0], dtype=np.float32)
    if sampled_values.size > target_nonzero_samples:
        sampled_values = rng.choice(sampled_values, size=target_nonzero_samples, replace=False)

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
    bin_edges = np.quantile(sampled_values, quantiles).astype(np.float32)
    np.save(cache_dir / "bin_edges.npy", bin_edges)

    neigh = NearestNeighbors(n_neighbors=knn_k + 1, algorithm="kd_tree")
    neigh.fit(coords)
    distances, indices = neigh.kneighbors(coords)
    distances = distances[:, 1:].astype(np.float32, copy=False)
    indices = indices[:, 1:].astype(np.int32, copy=False)
    np.save(cache_dir / "knn_indices.npy", indices)
    np.save(cache_dir / "knn_distances.npy", distances)

    gene_names = list(map(str, adata.var_names))
    (cache_dir / "gene_names.txt").write_text("\n".join(gene_names), encoding="utf-8")

    lr_pairs = load_lr_pairs(lr_db_path) if lr_db_path else []
    lr_pairs_idx, receptor_gene_ids = map_lr_pairs_to_genes(gene_names, lr_pairs)
    np.save(cache_dir / "lr_pairs_idx.npy", lr_pairs_idx)
    np.save(cache_dir / "receptor_gene_ids.npy", receptor_gene_ids)

    metadata = {
        "cache_metadata_version": CACHE_METADATA_VERSION,
        "h5ad_path": str(h5ad_path),
        "num_cells": int(num_cells),
        "num_genes": int(num_genes),
        "knn_k": int(knn_k),
        "num_bins": int(num_bins),
        "max_cells": int(max_cells),
        **_lr_db_metadata(lr_db_path),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return cache_dir


def load_slice_cache(cache_dir: str | Path) -> SliceCache:
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    expr = np.load(cache_dir / "expr_log1p_float16.npy", mmap_mode="r")
    coords = np.load(cache_dir / "coords.npy", mmap_mode="r")
    knn_indices = np.load(cache_dir / "knn_indices.npy", mmap_mode="r")
    knn_distances = np.load(cache_dir / "knn_distances.npy", mmap_mode="r")
    bin_edges = np.load(cache_dir / "bin_edges.npy")
    gene_names = (cache_dir / "gene_names.txt").read_text(encoding="utf-8").splitlines()
    lr_pairs_idx = np.load(cache_dir / "lr_pairs_idx.npy")
    receptor_gene_ids = np.load(cache_dir / "receptor_gene_ids.npy")
    return SliceCache(
        cache_dir=cache_dir,
        expr=expr,
        coords=coords,
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        bin_edges=bin_edges,
        gene_names=gene_names,
        lr_pairs_idx=lr_pairs_idx,
        receptor_gene_ids=receptor_gene_ids,
        num_cells=int(metadata["num_cells"]),
        num_genes=int(metadata["num_genes"]),
        nuclear_expr=(np.load(cache_dir / "nuclear_log1p_float16.npy", mmap_mode="r")
                      if metadata.get("compartment_cache_version") else None),
        cytoplasmic_expr=(np.load(cache_dir / "cytoplasmic_log1p_float16.npy", mmap_mode="r")
                         if metadata.get("compartment_cache_version") else None),
        bin_offset=int(metadata.get("bin_offset", 0)),
    )
