from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset

from spatial_pretrain.config import Config
from spatial_pretrain.data.cache import SliceCache, load_slice_cache
from spatial_pretrain.data.compartment_binning import configured_edges, positive_bins
from spatial_pretrain.data.lr import compute_center_candidate_lr_scores, compute_pairwise_lr_scores


@dataclass
class CenterSplit:
    train_indices: np.ndarray
    val_indices: np.ndarray


@dataclass
class PreparedSliceViews:
    train_centers_path: Path
    val_centers_path: Path
    train_nodes_path: Path
    val_nodes_path: Path
    train_epoch0_rows: np.ndarray
    val_rows: np.ndarray


VIEW_CACHE_VERSION = 2


def _axis_block_ids(values: np.ndarray, num_blocks: int) -> np.ndarray:
    if num_blocks <= 1:
        return np.zeros((values.shape[0],), dtype=np.int32)

    quantiles = np.linspace(0.0, 1.0, num_blocks + 1)
    edges = np.quantile(values, quantiles)
    inner_edges = np.asarray(edges[1:-1], dtype=np.float32)
    return np.searchsorted(inner_edges, values, side="right").astype(np.int32)


def _spatial_block_ids(coords: np.ndarray, blocks_x: int, blocks_y: int) -> np.ndarray:
    x_ids = _axis_block_ids(coords[:, 0], blocks_x)
    y_ids = _axis_block_ids(coords[:, 1], blocks_y)
    return x_ids + np.int32(max(blocks_x, 1)) * y_ids


def make_center_split(
    coords: np.ndarray,
    train_fraction: float,
    seed: int,
    *,
    strategy: str = "random",
    blocks_x: int = 1,
    blocks_y: int = 1,
) -> CenterSplit:
    num_cells = int(coords.shape[0])
    rng = np.random.default_rng(seed)
    indices = np.arange(num_cells, dtype=np.int64)
    train_size = min(max(int(round(num_cells * train_fraction)), 1), num_cells - 1)

    if strategy == "random":
        rng.shuffle(indices)
        return CenterSplit(
            train_indices=np.sort(indices[:train_size]),
            val_indices=np.sort(indices[train_size:]),
        )

    if strategy != "spatial_blocks":
        raise ValueError(f"Unsupported split_strategy={strategy!r}")

    block_ids = _spatial_block_ids(np.asarray(coords, dtype=np.float32), blocks_x, blocks_y)
    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)

    train_mask = np.zeros((num_cells,), dtype=bool)
    picked_blocks = 0
    for block_id in unique_blocks[:-1]:
        block_mask = block_ids == block_id
        if not block_mask.any():
            continue
        train_mask |= block_mask
        picked_blocks += 1
        if int(train_mask.sum()) >= train_size:
            break

    if picked_blocks == 0 or train_mask.all():
        rng.shuffle(indices)
        return CenterSplit(
            train_indices=np.sort(indices[:train_size]),
            val_indices=np.sort(indices[train_size:]),
        )

    train_indices = np.flatnonzero(train_mask).astype(np.int64)
    val_indices = np.flatnonzero(~train_mask).astype(np.int64)
    if val_indices.size == 0:
        rng.shuffle(indices)
        return CenterSplit(
            train_indices=np.sort(indices[:train_size]),
            val_indices=np.sort(indices[train_size:]),
        )

    return CenterSplit(
        train_indices=np.sort(train_indices),
        val_indices=np.sort(val_indices),
    )


def _view_cache_key(config: Config) -> str:
    payload = {
        "version": VIEW_CACHE_VERSION,
        "train_fraction": float(config.data.train_fraction),
        "split_seed": int(config.data.split_seed),
        "split_strategy": str(config.data.split_strategy),
        "split_blocks_x": int(config.data.split_blocks_x),
        "split_blocks_y": int(config.data.split_blocks_y),
        "restrict_subgraph_to_split": bool(config.data.restrict_subgraph_to_split),
        "center_selection_strategy": str(config.data.center_selection_strategy),
        "max_center_overlap_ratio": float(config.data.max_center_overlap_ratio),
        "max_cell_reuse": int(config.data.max_cell_reuse),
        "sampling_strategy": str(config.data.sampling_strategy),
        "subgraph_size": int(config.data.subgraph_size),
        "nearest_k": int(config.data.nearest_k),
        "candidate_k": int(config.data.candidate_k),
        "lr_k": int(config.data.lr_k),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


def _prepared_root(cache_dir: Path, config: Config) -> Path:
    root = cache_dir / "prepared_views" / _view_cache_key(config)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_prepared_views_if_valid(
    *,
    meta_path: Path,
    train_centers_path: Path,
    val_centers_path: Path,
    train_nodes_path: Path,
    val_nodes_path: Path,
    train_epoch0_rows_path: Path,
    val_rows_path: Path,
    config: Config,
    slice_name: str,
) -> PreparedSliceViews | None:
    paths = (
        train_centers_path,
        val_centers_path,
        train_nodes_path,
        val_nodes_path,
        train_epoch0_rows_path,
        val_rows_path,
    )
    if not meta_path.exists() and not any(path.exists() for path in paths):
        return None
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Prepared view cache {meta_path.parent} is incomplete; missing files: {missing}")
    if not meta_path.exists():
        raise RuntimeError(f"Prepared view cache {meta_path.parent} is missing metadata.json")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_key = _view_cache_key(config)
    mismatches: list[str] = []
    if int(metadata.get("view_cache_version", -1)) != VIEW_CACHE_VERSION:
        mismatches.append(
            f"view_cache_version metadata={metadata.get('view_cache_version')!r} expected={VIEW_CACHE_VERSION!r}"
        )
    if metadata.get("cache_key") != expected_key:
        mismatches.append(f"cache_key metadata={metadata.get('cache_key')!r} expected={expected_key!r}")
    if metadata.get("slice_name") != slice_name:
        mismatches.append(f"slice_name metadata={metadata.get('slice_name')!r} expected={slice_name!r}")
    if int(metadata.get("subgraph_size", -1)) != int(config.data.subgraph_size):
        mismatches.append(
            f"subgraph_size metadata={metadata.get('subgraph_size')!r} expected={int(config.data.subgraph_size)!r}"
        )

    train_centers = np.load(train_centers_path, mmap_mode="r")
    val_centers = np.load(val_centers_path, mmap_mode="r")
    train_nodes = np.load(train_nodes_path, mmap_mode="r")
    val_nodes = np.load(val_nodes_path, mmap_mode="r")
    train_epoch0_rows = np.load(train_epoch0_rows_path)
    val_rows = np.load(val_rows_path)
    subgraph_size = int(config.data.subgraph_size)
    if train_nodes.shape != (len(train_centers), subgraph_size):
        mismatches.append(f"train_nodes shape={train_nodes.shape!r} expected={(len(train_centers), subgraph_size)!r}")
    if val_nodes.shape != (len(val_centers), subgraph_size):
        mismatches.append(f"val_nodes shape={val_nodes.shape!r} expected={(len(val_centers), subgraph_size)!r}")
    if train_epoch0_rows.size and int(train_epoch0_rows.max()) >= len(train_centers):
        mismatches.append("train_epoch0_rows contains row ids outside train_centers")
    if val_rows.size and int(val_rows.max()) >= len(val_centers):
        mismatches.append("val_rows contains row ids outside val_centers")
    for key, value in (
        ("train_centers", len(train_centers)),
        ("val_centers", len(val_centers)),
        ("train_epoch0_rows", len(train_epoch0_rows)),
        ("val_rows", len(val_rows)),
    ):
        if int(metadata.get(key, -1)) != int(value):
            mismatches.append(f"{key} metadata={metadata.get(key)!r} expected={int(value)!r}")
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(f"Prepared view cache {meta_path.parent} does not match the requested config: {joined}")

    print(f"[dataset-cache] {slice_name} loaded prepared views from {meta_path.parent.name}", flush=True)
    return PreparedSliceViews(
        train_centers_path=train_centers_path,
        val_centers_path=val_centers_path,
        train_nodes_path=train_nodes_path,
        val_nodes_path=val_nodes_path,
        train_epoch0_rows=train_epoch0_rows,
        val_rows=val_rows,
    )


def _rng_for_center(config: Config, *, training: bool, epoch: int, center_idx: int) -> np.random.Generator:
    base = config.data.split_seed if training else config.data.split_seed + 10_000_019
    return np.random.default_rng(base + epoch * 1_000_003 + int(center_idx))


def _make_allowed_node_mask(
    num_cells: int,
    center_indices: np.ndarray,
    restrict_subgraph_to_split: bool,
) -> np.ndarray:
    mask = np.ones((num_cells,), dtype=bool)
    if restrict_subgraph_to_split:
        mask[:] = False
        mask[np.asarray(center_indices, dtype=np.int64)] = True
    return mask


def _candidate_nodes_from_cache(
    cache: SliceCache,
    config: Config,
    center_idx: int,
    rng: np.random.Generator,
    allowed_node_mask: np.ndarray,
) -> np.ndarray:
    k_total = config.data.subgraph_size - 1
    candidates = np.asarray(cache.knn_indices[center_idx][: config.data.candidate_k], dtype=np.int64)
    candidates = candidates[candidates != center_idx]
    if config.data.restrict_subgraph_to_split:
        candidates = candidates[allowed_node_mask[candidates]]

    nearest = candidates[: config.data.nearest_k]
    selected: list[int] = list(map(int, nearest))

    if config.data.sampling_strategy == "one_hop":
        if len(selected) < k_total:
            for node in candidates:
                node = int(node)
                if node not in selected:
                    selected.append(node)
                if len(selected) >= k_total:
                    break
        if len(selected) < k_total:
            pad = [selected[-1] if selected else int(center_idx)] * (k_total - len(selected))
            selected.extend(pad)
        return np.asarray([center_idx] + selected[:k_total], dtype=np.int64)

    remaining = [int(x) for x in candidates if int(x) not in selected]
    if config.data.sampling_strategy == "mixed" and config.data.lr_k > 0 and remaining:
        center_expr = np.asarray(cache.expr[center_idx], dtype=np.float32)
        candidate_expr = np.asarray(cache.expr[np.asarray(remaining, dtype=np.int64)], dtype=np.float32)
        lr_scores = compute_center_candidate_lr_scores(center_expr, candidate_expr, cache.lr_pairs_idx)
        order = np.argsort(-lr_scores)
        for idx in order[: config.data.lr_k]:
            node = remaining[int(idx)]
            if node not in selected:
                selected.append(node)

    for node in candidates:
        node = int(node)
        if node not in selected:
            selected.append(node)
        if len(selected) >= k_total:
            break

    if len(selected) < k_total:
        pad = [selected[-1] if selected else int(center_idx)] * (k_total - len(selected))
        selected.extend(pad)
    return np.asarray([center_idx] + selected[:k_total], dtype=np.int64)


def _build_subgraph_table(
    cache: SliceCache,
    center_indices: np.ndarray,
    config: Config,
    allowed_node_mask: np.ndarray,
    output_path: Path,
) -> Path:
    if output_path.exists():
        return output_path

    subgraph_size = int(config.data.subgraph_size)
    staged_path = output_path.with_suffix(output_path.suffix + ".partial")
    nodes_memmap = open_memmap(
        staged_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(center_indices), subgraph_size),
    )
    centers = np.asarray(center_indices, dtype=np.int64)
    if config.data.sampling_strategy == "one_hop":
        # Same ordered neighbors and last-neighbor padding as the scalar path.
        # Batch it for multi-million-cell slices instead of constructing an RNG
        # and Python lists for every center.
        for start in range(0, len(centers), 8192):
            block = centers[start:start+8192]
            candidates = np.asarray(cache.knn_indices[block, :config.data.candidate_k], dtype=np.int32)
            valid = candidates != block[:, None]
            if config.data.restrict_subgraph_to_split:
                valid &= allowed_node_mask[candidates]
            count = valid.sum(axis=1)
            order = np.argsort(~valid, axis=1, kind="stable")
            ordered = np.take_along_axis(candidates, order, axis=1)
            positions = np.minimum(np.arange(subgraph_size - 1)[None, :], np.maximum(count - 1, 0)[:, None])
            if candidates.shape[1]:
                neighbors = np.take_along_axis(ordered, positions, axis=1)
                neighbors[count == 0] = block[count == 0, None]
            else:
                neighbors = np.broadcast_to(block[:, None], (len(block), subgraph_size - 1))
            nodes_memmap[start:start+len(block), 0] = block
            nodes_memmap[start:start+len(block), 1:] = neighbors
    else:
        for row_idx, center_idx in enumerate(centers):
            rng = _rng_for_center(config, training=True, epoch=0, center_idx=int(center_idx))
            nodes = _candidate_nodes_from_cache(cache, config, int(center_idx), rng, allowed_node_mask)
            nodes_memmap[row_idx] = nodes.astype(np.int32, copy=False)
    nodes_memmap.flush()
    del nodes_memmap
    staged_path.replace(output_path)
    return output_path


def _select_low_overlap_rows(
    center_indices: np.ndarray,
    subgraph_nodes: np.ndarray,
    config: Config,
    *,
    training: bool,
    epoch: int,
    num_cells: int,
    slice_name: str,
) -> np.ndarray:
    if config.data.center_selection_strategy == "all":
        return np.arange(len(center_indices), dtype=np.int64)
    if config.data.center_selection_strategy != "low_overlap":
        raise ValueError(
            f"Unsupported center_selection_strategy={config.data.center_selection_strategy!r}"
        )

    order = np.arange(len(center_indices), dtype=np.int64)
    seed = (
        config.data.split_seed + epoch * 10_000_019
        if training
        else config.data.split_seed + 20_000_033
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    usage = np.zeros((num_cells,), dtype=np.uint16)
    selected_rows: list[int] = []
    max_reuse = int(config.data.max_cell_reuse)
    max_overlap = float(config.data.max_center_overlap_ratio)

    for row_idx in order:
        unique_nodes = np.unique(np.asarray(subgraph_nodes[row_idx], dtype=np.int64))
        if max_reuse > 0 and np.any(usage[unique_nodes] >= max_reuse):
            continue
        if max_overlap < 1.0:
            overlap_ratio = float((usage[unique_nodes] > 0).mean())
            if overlap_ratio > max_overlap:
                continue
        selected_rows.append(int(row_idx))
        usage[unique_nodes] += 1

    if not selected_rows:
        raise ValueError("low_overlap center selection rejected every candidate center")

    selected_rows_arr = np.sort(np.asarray(selected_rows, dtype=np.int64))
    phase = "train" if training else "val"
    print(
        f"[dataset] {slice_name} {phase} centers reduced from {len(center_indices)} to {len(selected_rows_arr)} "
        f"with strategy=low_overlap max_reuse={max_reuse} "
        f"max_overlap={max_overlap:.3f} restrict_to_split={config.data.restrict_subgraph_to_split}",
        flush=True,
    )
    return selected_rows_arr


def prepare_slice_views(cache: SliceCache, config: Config, slice_name: str) -> PreparedSliceViews:
    import fcntl
    root = _prepared_root(cache.cache_dir, config)
    with (root / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_slice_views_unlocked(cache, config, slice_name)


def _prepare_slice_views_unlocked(cache: SliceCache, config: Config, slice_name: str) -> PreparedSliceViews:
    root = _prepared_root(cache.cache_dir, config)
    meta_path = root / "metadata.json"
    train_centers_path = root / "train_centers.npy"
    val_centers_path = root / "val_centers.npy"
    train_nodes_path = root / "train_nodes.npy"
    val_nodes_path = root / "val_nodes.npy"
    train_epoch0_rows_path = root / "train_epoch0_rows.npy"
    val_rows_path = root / "val_rows.npy"

    cached = _load_prepared_views_if_valid(
        meta_path=meta_path,
        train_centers_path=train_centers_path,
        val_centers_path=val_centers_path,
        train_nodes_path=train_nodes_path,
        val_nodes_path=val_nodes_path,
        train_epoch0_rows_path=train_epoch0_rows_path,
        val_rows_path=val_rows_path,
        config=config,
        slice_name=slice_name,
    )
    if cached is not None:
        return cached

    split = make_center_split(
        cache.coords,
        config.data.train_fraction,
        config.data.split_seed,
        strategy=config.data.split_strategy,
        blocks_x=config.data.split_blocks_x,
        blocks_y=config.data.split_blocks_y,
    )
    train_centers = np.sort(np.asarray(split.train_indices, dtype=np.int64))
    val_centers = np.sort(np.asarray(split.val_indices, dtype=np.int64))
    np.save(train_centers_path, train_centers)
    np.save(val_centers_path, val_centers)

    train_allowed_mask = _make_allowed_node_mask(
        cache.num_cells,
        train_centers,
        config.data.restrict_subgraph_to_split,
    )
    val_allowed_mask = _make_allowed_node_mask(
        cache.num_cells,
        val_centers,
        config.data.restrict_subgraph_to_split,
    )

    _build_subgraph_table(cache, train_centers, config, train_allowed_mask, train_nodes_path)
    _build_subgraph_table(cache, val_centers, config, val_allowed_mask, val_nodes_path)

    train_nodes = np.load(train_nodes_path, mmap_mode="r")
    val_nodes = np.load(val_nodes_path, mmap_mode="r")
    train_epoch0_rows = _select_low_overlap_rows(
        train_centers,
        train_nodes,
        config,
        training=True,
        epoch=0,
        num_cells=cache.num_cells,
        slice_name=slice_name,
    )
    val_rows = _select_low_overlap_rows(
        val_centers,
        val_nodes,
        config,
        training=False,
        epoch=0,
        num_cells=cache.num_cells,
        slice_name=slice_name,
    )
    np.save(train_epoch0_rows_path, train_epoch0_rows)
    np.save(val_rows_path, val_rows)

    metadata = {
        "view_cache_version": VIEW_CACHE_VERSION,
        "cache_key": _view_cache_key(config),
        "slice_name": slice_name,
        "subgraph_size": int(config.data.subgraph_size),
        "train_centers": int(len(train_centers)),
        "val_centers": int(len(val_centers)),
        "train_epoch0_rows": int(len(train_epoch0_rows)),
        "val_rows": int(len(val_rows)),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[dataset-cache] {slice_name} saved prepared views to {root.name}", flush=True)

    return PreparedSliceViews(
        train_centers_path=train_centers_path,
        val_centers_path=val_centers_path,
        train_nodes_path=train_nodes_path,
        val_nodes_path=val_nodes_path,
        train_epoch0_rows=train_epoch0_rows,
        val_rows=val_rows,
    )


def _prepare_slice_views_worker(cache_dir: str, config: Config, slice_name: str) -> PreparedSliceViews:
    cache = load_slice_cache(cache_dir)
    return prepare_slice_views(cache, config, slice_name)


def prepare_slice_views_parallel(
    caches: list[SliceCache],
    slice_names: list[str],
    config: Config,
) -> list[PreparedSliceViews]:
    max_workers = max(int(config.data.view_prep_workers), 0)
    if max_workers <= 1 or len(caches) <= 1:
        return [prepare_slice_views(cache, config, slice_name) for cache, slice_name in zip(caches, slice_names)]

    results: list[PreparedSliceViews | None] = [None] * len(caches)
    with ProcessPoolExecutor(
        max_workers=min(max_workers, len(caches)),
        mp_context=mp.get_context("spawn"),
    ) as executor:
        future_to_index = {
            executor.submit(_prepare_slice_views_worker, str(cache.cache_dir), config, slice_name): index
            for index, (cache, slice_name) in enumerate(zip(caches, slice_names))
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return [result for result in results if result is not None]


def _resample_rows_worker(
    nodes_path: str,
    centers_path: str,
    config: Config,
    training: bool,
    slice_name: str,
    epoch: int,
    num_cells: int,
) -> np.ndarray:
    nodes = np.load(nodes_path, mmap_mode="r")
    center_indices = np.load(centers_path, mmap_mode="r")
    return _select_low_overlap_rows(
        center_indices,
        nodes,
        config,
        training=training,
        epoch=epoch,
        num_cells=num_cells,
        slice_name=slice_name,
    )


class SpatialSubgraphDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        subgraph_nodes_path: str | Path,
        center_indices_path: str | Path,
        selected_rows: np.ndarray,
        config: Config,
        *,
        training: bool,
        global_gene_ids: np.ndarray | None = None,
        pad_gene_id: int | None = None,
        slice_name: str = "",
        slice_cache: SliceCache | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.subgraph_nodes_path = Path(subgraph_nodes_path)
        self.center_indices_path = Path(center_indices_path)
        self.config = config
        self.training = training
        self.epoch = 0
        self._cache: SliceCache | None = slice_cache
        self._subgraph_nodes = None
        self._center_indices = None
        cache = self._ensure_cache()
        self.gene_sampling_strategy = str(config.data.gene_sampling_strategy).strip().lower()
        if self.gene_sampling_strategy not in {"top", "random_nonzero", "compartment_balanced"}:
            raise ValueError(f"Unsupported gene_sampling_strategy={config.data.gene_sampling_strategy!r}")
        self.top_genes = (
            int(config.data.top_genes)
            if self.gene_sampling_strategy in {"random_nonzero", "compartment_balanced"}
            else min(int(config.data.top_genes), cache.num_genes)
        )
        self.mask_token_id = config.data.num_bins
        self.pad_value_id = config.data.num_bins + 1
        self.global_gene_ids = (
            np.asarray(global_gene_ids, dtype=np.int64)
            if global_gene_ids is not None
            else np.arange(cache.num_genes, dtype=np.int64)
        )
        max_gene_id = int(self.global_gene_ids.max()) if self.global_gene_ids.size else -1
        self.pad_gene_id = int(pad_gene_id) if pad_gene_id is not None else max_gene_id + 1
        self.slice_name = slice_name or self.cache_dir.name
        self.selected_rows = np.asarray(selected_rows, dtype=np.int64)
        if self.gene_sampling_strategy == "compartment_balanced":
            if config.data.nuclear_tokens <= 0 or config.data.cytoplasmic_tokens <= 0:
                raise ValueError("Both compartment token budgets must be positive")
            if self.top_genes != config.data.nuclear_tokens + config.data.cytoplasmic_tokens:
                raise ValueError("top_genes must equal nuclear_tokens + cytoplasmic_tokens")
            if cache.nuclear_expr is None or cache.cytoplasmic_expr is None:
                raise ValueError("compartment_balanced requires a compartment cache")
            if config.data.use_lr_prior:
                raise ValueError("Expression-derived LR bias is disabled for compartment training")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = None
        state["_subgraph_nodes"] = None
        state["_center_indices"] = None
        return state

    def _ensure_cache(self) -> SliceCache:
        if self._cache is None:
            self._cache = load_slice_cache(self.cache_dir)
        return self._cache

    def _ensure_subgraph_nodes(self) -> np.ndarray:
        if self._subgraph_nodes is None:
            self._subgraph_nodes = np.load(self.subgraph_nodes_path, mmap_mode="r")
        return self._subgraph_nodes

    def _ensure_center_indices(self) -> np.ndarray:
        if self._center_indices is None:
            self._center_indices = np.load(self.center_indices_path, mmap_mode="r")
        return self._center_indices

    @property
    def num_cells(self) -> int:
        return self._ensure_cache().num_cells

    def __len__(self) -> int:
        return len(self.selected_rows)

    def set_epoch(self, epoch: int) -> None:
        if epoch == self.epoch:
            return
        self.epoch = epoch
        if self.training and self.config.data.resample_views_each_epoch:
            self.selected_rows = _resample_rows_worker(
                str(self.subgraph_nodes_path),
                str(self.center_indices_path),
                self.config,
                True,
                self.slice_name,
                epoch,
                self.num_cells,
            )

    def _rng(self, center_idx: int) -> np.random.Generator:
        return _rng_for_center(self.config, training=self.training, epoch=self.epoch, center_idx=center_idx)

    def _select_gene_tokens(
        self,
        expr_subgraph: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_cells, num_genes = expr_subgraph.shape
        if self.gene_sampling_strategy == "top":
            order = np.argsort(-expr_subgraph, axis=1)[:, : self.top_genes].astype(np.int64, copy=False)
            values = np.take_along_axis(expr_subgraph, order, axis=1)
            bins = np.digitize(values, self._ensure_cache().bin_edges, right=False).astype(np.int64)
            token_valid_mask = np.ones_like(bins, dtype=bool)
            return order, bins, token_valid_mask

        local_gene_ids = np.zeros((num_cells, self.top_genes), dtype=np.int64)
        value_bins = np.zeros((num_cells, self.top_genes), dtype=np.int64)
        token_valid_mask = np.zeros((num_cells, self.top_genes), dtype=bool)
        bin_edges = self._ensure_cache().bin_edges

        for row_idx in range(num_cells):
            expressed = np.flatnonzero(expr_subgraph[row_idx] > 0)
            if expressed.size == 0:
                continue
            if expressed.size > self.top_genes:
                selected = rng.choice(expressed, size=self.top_genes, replace=False)
                # Keep token positions stable after stochastic inclusion sampling.
                selected = np.sort(selected)
            else:
                selected = expressed
            token_count = int(selected.size)
            local_gene_ids[row_idx, :token_count] = selected.astype(np.int64, copy=False)
            values = expr_subgraph[row_idx, selected]
            value_bins[row_idx, :token_count] = np.digitize(values, bin_edges, right=False).astype(np.int64)
            token_valid_mask[row_idx, :token_count] = True

        return local_gene_ids, value_bins, token_valid_mask

    def _apply_masks(
        self,
        gene_ids: np.ndarray,
        value_bins: np.ndarray,
        token_valid_mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_bins = value_bins.copy()
        uniform_mask_ratio = float(self.config.masking.uniform_mask_ratio)
        if uniform_mask_ratio >= 0:
            loss_mask = (rng.random(value_bins.shape) < uniform_mask_ratio) & token_valid_mask
            valid_cells = token_valid_mask.any(axis=1)
            empty_cells = valid_cells & ~loss_mask.any(axis=1)
            if empty_cells.any():
                first_valid = token_valid_mask[empty_cells].argmax(axis=1)
                loss_mask[np.flatnonzero(empty_cells), first_valid] = True
            input_bins[loss_mask] = self.mask_token_id
            input_bins[~token_valid_mask] = self.pad_value_id
            center_mask = np.zeros((value_bins.shape[1],), dtype=bool)
            return input_bins, loss_mask, center_mask

        loss_mask = np.zeros_like(value_bins, dtype=bool)
        center_mask = np.zeros((value_bins.shape[1],), dtype=bool)

        neighbor_mask = (
            rng.random(value_bins[1:].shape) < self.config.masking.general_mask_ratio
        ) & token_valid_mask[1:]
        loss_mask[1:] = neighbor_mask
        input_bins[1:][neighbor_mask] = self.mask_token_id

        receptor_gene_ids = self._ensure_cache().receptor_gene_ids
        center_gene_ids = gene_ids[0]
        receptor_mask = np.isin(center_gene_ids, receptor_gene_ids) & token_valid_mask[0]
        receptor_keep = receptor_mask & (
            rng.random(center_gene_ids.shape[0]) < self.config.masking.receptor_keep_ratio
        )
        candidate_center_mask = token_valid_mask[0] & ~receptor_keep
        center_mask = (
            rng.random(center_gene_ids.shape[0]) < self.config.masking.center_mask_ratio
        ) & candidate_center_mask
        if not center_mask.any() and candidate_center_mask.any():
            first = int(np.flatnonzero(candidate_center_mask)[0])
            center_mask[first] = True

        loss_mask[0] = center_mask
        input_bins[0][center_mask] = self.mask_token_id
        input_bins[~token_valid_mask] = self.pad_value_id
        return input_bins, loss_mask, center_mask

    def _select_compartment_tokens(self, node_indices, rng):
        """Independent nonzero sampling, with a fixed block for each compartment."""
        cache = self._ensure_cache()
        budgets = (self.config.data.nuclear_tokens, self.config.data.cytoplasmic_tokens)
        edges = configured_edges(self.config.data)
        if edges is None:
            edges = cache.bin_edges
        ids = np.zeros((len(node_indices), sum(budgets)), dtype=np.int64)
        bins = np.zeros_like(ids)
        valid = np.zeros_like(ids, dtype=bool)
        compartments = np.full_like(ids, 2)  # 0=nuclear, 1=cytoplasmic, 2=PAD
        offset = 0
        for compartment, (expr, budget) in enumerate(zip(
            (cache.nuclear_expr, cache.cytoplasmic_expr), budgets
        )):
            for row, node in enumerate(node_indices):
                values = np.asarray(expr[node], dtype=np.float32)
                expressed = np.flatnonzero(values > 0)
                if len(expressed) > budget:
                    expressed = np.sort(rng.choice(expressed, size=budget, replace=False))
                slots = slice(offset, offset + len(expressed))
                ids[row, slots] = expressed
                bins[row, slots] = positive_bins(values[expressed], edges)
                valid[row, slots] = True
                compartments[row, slots] = compartment
            offset += budget
        return ids, bins, valid, compartments

    def _build_lr_tensors(self, expr_subgraph: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lr_pairs_idx = self._ensure_cache().lr_pairs_idx
        sender_to_receiver = compute_pairwise_lr_scores(expr_subgraph, lr_pairs_idx)
        receiver_to_sender = sender_to_receiver.T
        row_has_signal = receiver_to_sender.sum(axis=-1, keepdims=True) > 0
        lr_logits = receiver_to_sender / max(self.config.data.lr_temperature, 1e-6)
        lr_logits = lr_logits - lr_logits.max(axis=-1, keepdims=True)
        prior = np.exp(lr_logits)
        prior = prior / (prior.sum(axis=-1, keepdims=True) + 1e-8)
        prior = np.where(row_has_signal, prior, 0.0)
        bias = np.where(row_has_signal, np.log(prior + 1e-8), 0.0)
        return bias.astype(np.float32), prior.astype(np.float32)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row_idx = int(self.selected_rows[item])
        center_idx = int(self._ensure_center_indices()[row_idx])
        node_indices = np.asarray(self._ensure_subgraph_nodes()[row_idx], dtype=np.int64)
        cache = self._ensure_cache()
        rng = self._rng(center_idx)

        expr_subgraph = np.asarray(cache.expr[node_indices], dtype=np.float32)
        coords_subgraph = np.asarray(cache.coords[node_indices], dtype=np.float32)
        compartment_ids = None
        if self.gene_sampling_strategy == "compartment_balanced":
            local_gene_ids, value_bins, token_valid_mask, compartment_ids = self._select_compartment_tokens(node_indices, rng)
        else:
            local_gene_ids, value_bins, token_valid_mask = self._select_gene_tokens(expr_subgraph, rng)
        input_bins, loss_mask, center_loss_mask = self._apply_masks(
            local_gene_ids,
            value_bins,
            token_valid_mask,
            rng,
        )
        if self.config.data.use_lr_prior:
            lr_bias, lr_prior = self._build_lr_tensors(expr_subgraph)
        else:
            lr_bias = lr_prior = np.zeros((len(node_indices), len(node_indices)), dtype=np.float32)
        gene_ids = self.global_gene_ids[local_gene_ids]
        gene_ids = np.where(token_valid_mask, gene_ids, self.pad_gene_id)

        cell_roles = np.zeros((len(node_indices),), dtype=np.int64)
        cell_roles[0] = 1

        result = {
            "center_index": torch.tensor(0, dtype=torch.long),
            "global_center_id": torch.tensor(center_idx, dtype=torch.long),
            "node_indices": torch.from_numpy(node_indices.astype(np.int64)),
            "gene_ids": torch.from_numpy(gene_ids),
            "input_bins": torch.from_numpy(input_bins.astype(np.int64)),
            "target_bins": torch.from_numpy(value_bins.astype(np.int64)),
            "loss_mask": torch.from_numpy(loss_mask),
            "center_loss_mask": torch.from_numpy(center_loss_mask),
            "token_valid_mask": torch.from_numpy(token_valid_mask),
            "cell_roles": torch.from_numpy(cell_roles),
            "coords": torch.from_numpy(coords_subgraph),
            "lr_bias": torch.from_numpy(lr_bias),
            "lr_prior": torch.from_numpy(lr_prior),
        }
        if compartment_ids is not None:
            result["compartment_ids"] = torch.from_numpy(compartment_ids)
        return result


class MultiSliceDataset(Dataset):
    def __init__(self, datasets: list[SpatialSubgraphDataset]) -> None:
        if not datasets:
            raise ValueError("MultiSliceDataset requires at least one child dataset")
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets], dtype=np.int64)

    def __len__(self) -> int:
        return int(self.cumulative_sizes[-1])

    def set_epoch(self, epoch: int) -> None:
        if all(dataset.epoch == epoch for dataset in self.datasets):
            return

        parallel_workers = max(int(self.datasets[0].config.data.view_prep_workers), 0)
        if (
            parallel_workers > 1
            and len(self.datasets) > 1
            and all(dataset.training for dataset in self.datasets)
            and self.datasets[0].config.data.resample_views_each_epoch
        ):
            with ProcessPoolExecutor(
                max_workers=min(parallel_workers, len(self.datasets)),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                future_to_index = {
                    executor.submit(
                        _resample_rows_worker,
                        str(dataset.subgraph_nodes_path),
                        str(dataset.center_indices_path),
                        dataset.config,
                        dataset.training,
                        dataset.slice_name,
                        epoch,
                        dataset.num_cells,
                    ): index
                    for index, dataset in enumerate(self.datasets)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    self.datasets[index].selected_rows = future.result()
                    self.datasets[index].epoch = epoch
        else:
            for dataset in self.datasets:
                dataset.set_epoch(epoch)
        self.cumulative_sizes = np.cumsum([len(ds) for ds in self.datasets], dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        prev_size = 0 if dataset_idx == 0 else int(self.cumulative_sizes[dataset_idx - 1])
        sample_idx = index - prev_size
        return self.datasets[dataset_idx][sample_idx]


def collate_subgraphs(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys}
