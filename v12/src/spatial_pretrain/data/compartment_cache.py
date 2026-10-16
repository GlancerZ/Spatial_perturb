from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
from numpy.lib.format import open_memmap
from sklearn.neighbors import NearestNeighbors

from spatial_pretrain.data.cache import _to_dense_float32, _validate_existing_cache
from spatial_pretrain.data.dataset import make_center_split


def build_compartment_cache(h5ad_path, cache_dir, config):
    """Build independent compartment caches with train-only shared bin edges."""
    source, destination = Path(h5ad_path), Path(cache_dir)
    cfg = config.data
    if cfg.num_bins < 3:
        raise ValueError("Compartment binning needs a zero bin and at least two positive bins")
    source_stat = source.stat()
    signature = {
        "compartment_cache_version": 1,
        "source_size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns,
        "split_seed": cfg.split_seed, "train_fraction": cfg.train_fraction,
        "split_strategy": cfg.split_strategy, "split_blocks_x": cfg.split_blocks_x,
        "split_blocks_y": cfg.split_blocks_y,
    }
    if destination.exists():
        _validate_existing_cache(cache_dir=destination, h5ad_path=source, lr_db_path="",
                                 knn_k=cfg.knn_k, num_bins=cfg.num_bins, max_cells=cfg.max_cells)
        metadata = json.loads((destination / "metadata.json").read_text())
        for key, value in signature.items():
            if metadata.get(key) != value:
                raise ValueError(f"Compartment cache mismatch: {key}; use a new cache directory")
        for name in ("nuclear_log1p_float16.npy", "cytoplasmic_log1p_float16.npy"):
            array = np.load(destination / name, mmap_mode="r")
            if array.shape != (metadata["num_cells"], metadata["num_genes"]):
                raise ValueError(f"Invalid compartment cache array {name}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
    adata = None
    try:
        adata = ad.read_h5ad(source, backed="r")
        for name in ("nuclear_counts", "extranuclear_counts"):
            if name not in adata.layers:
                raise ValueError(f"Missing layer {name} in {source}; run transcript aggregation first")
        cells = min(adata.n_obs, cfg.max_cells) if cfg.max_cells > 0 else adata.n_obs
        genes = adata.n_vars
        if cells <= cfg.knn_k:
            raise ValueError(f"Need more than knn_k={cfg.knn_k} cells, got {cells}")
        coords = np.asarray(adata.obsm["spatial"][:cells], dtype=np.float32)
        if coords.shape != (cells, 2) or not np.isfinite(coords).all():
            raise ValueError("Expected finite 2D spatial coordinates")
        split = make_center_split(coords, cfg.train_fraction, cfg.split_seed,
                                  strategy=cfg.split_strategy, blocks_x=cfg.split_blocks_x,
                                  blocks_y=cfg.split_blocks_y)
        train = np.zeros(cells, dtype=bool)
        train[split.train_indices] = True
        np.save(staging / "coords.npy", coords)
        filenames = ("nuclear_log1p_float16.npy", "cytoplasmic_log1p_float16.npy", "expr_log1p_float16.npy")
        maps = [open_memmap(staging / name, mode="w+", dtype=np.float16, shape=(cells, genes))
                for name in filenames]
        rng = np.random.default_rng(cfg.split_seed)
        samples = []
        for start in range(0, cells, cfg.chunk_size):
            stop = min(start + cfg.chunk_size, cells)
            raw = [_to_dense_float32(adata.layers[name][start:stop])
                   for name in ("nuclear_counts", "extranuclear_counts")]
            if any(not np.isfinite(x).all() or (x < 0).any() or (x != np.floor(x)).any() for x in raw):
                raise ValueError("Compartment layers must contain finite nonnegative integer counts")
            total = raw[0] + raw[1]
            if not np.array_equal(total, _to_dense_float32(adata.X[start:stop])):
                raise ValueError("nuclear_counts + extranuclear_counts must equal X")
            logged = [np.log1p(x) for x in (*raw, total)]
            for array, values in zip(maps, logged):
                array[start:stop] = values
            # Sample the same float16-represented values consumed by the dataset.
            positive = np.concatenate([x[train[start:stop]].astype(np.float16).astype(np.float32).ravel()
                                       for x in logged[:2]])
            positive = positive[positive > 0]
            if len(positive):
                samples.append(rng.choice(positive, min(len(positive), 50_000), replace=False))
            print(f"[compartment-cache] {source.stem} rows={stop}/{cells}", flush=True)
        for array in maps:
            array.flush()
        del maps
        positive = np.concatenate(samples) if samples else np.array([1.0], dtype=np.float32)
        if len(positive) > 2_000_000:
            positive = rng.choice(positive, 2_000_000, replace=False)
        # 0 is reserved for real zero; bins 1..num_bins-1 encode positive values.
        edges = np.quantile(positive, np.linspace(0, 1, cfg.num_bins)[1:-1]).astype(np.float32)
        np.save(staging / "bin_edges.npy", edges)
        nn = NearestNeighbors(n_neighbors=cfg.knn_k + 1, algorithm="kd_tree").fit(coords)
        distances, indices = nn.kneighbors(coords)
        # Remove each actual query ID, including coordinate ties.
        neighbors = np.empty((cells, cfg.knn_k), dtype=np.int32)
        neighbor_distances = np.empty((cells, cfg.knn_k), dtype=np.float32)
        for i in range(cells):
            keep = indices[i] != i
            neighbors[i] = indices[i][keep][:cfg.knn_k]
            neighbor_distances[i] = distances[i][keep][:cfg.knn_k]
        np.save(staging / "knn_indices.npy", neighbors)
        np.save(staging / "knn_distances.npy", neighbor_distances)
        np.save(staging / "lr_pairs_idx.npy", np.empty((0, 2), dtype=np.int32))
        np.save(staging / "receptor_gene_ids.npy", np.empty(0, dtype=np.int32))
        (staging / "gene_names.txt").write_text("\n".join(map(str, adata.var_names)))
        metadata = dict(signature, h5ad_path=str(source.resolve()), num_cells=cells, num_genes=genes,
                        knn_k=cfg.knn_k, num_bins=cfg.num_bins, max_cells=cfg.max_cells,
                        lr_db_path="", lr_db_size=0, lr_db_mtime_ns=0, bin_offset=1,
                        bin_fit_cells=len(split.train_indices))
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))
        staging.rename(destination)
        return destination
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        if adata is not None:
            adata.file.close()
