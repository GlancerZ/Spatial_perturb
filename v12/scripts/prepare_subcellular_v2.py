#!/usr/bin/env python3
"""Build resumable V2 raw-count caches; run on a Slurm compute node."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import errno
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from numpy.lib.format import open_memmap
from scipy import sparse
import yaml

from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA, stable_seed, vocabulary_path


def read_matrix_rows(element, start, stop, total_genes):
    """Read only requested H5AD rows, including backed CSR layers."""
    if isinstance(element, h5py.Dataset):
        return np.asarray(element[start:stop], dtype=np.float32)
    encoding = element.attrs.get("encoding-type", "")
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if encoding != "csr_matrix":
        raise ValueError(f"Expected dense or CSR count layer, got {encoding!r}; convert the source losslessly first")
    ptr = np.asarray(element["indptr"][start:stop + 1], dtype=np.int64)
    first, last = int(ptr[0]), int(ptr[-1])
    matrix = sparse.csr_matrix((
        np.asarray(element["data"][first:last], dtype=np.float32),
        np.asarray(element["indices"][first:last], dtype=np.int32), ptr - first,
    ), shape=(stop - start, total_genes))
    return matrix.toarray()


def spatial_split(coords, data):
    """Hold out entire coordinate quantile blocks, without accessing labels."""
    ids = []
    for axis, key in enumerate(("split_blocks_x", "split_blocks_y")):
        edges = np.quantile(coords[:, axis], np.linspace(0, 1, int(data[key]) + 1))
        ids.append(np.searchsorted(edges[1:-1], coords[:, axis], side="right"))
    blocks = ids[0] + int(data["split_blocks_x"]) * ids[1]
    unique = np.unique(blocks)
    if len(unique) < 2:
        raise ValueError("Spatial validation requires at least two nonempty coordinate blocks")
    rng = np.random.default_rng(int(data["seed"]))
    rng.shuffle(unique)
    train = np.zeros(len(coords), dtype=bool)
    target = max(1, int(round(len(coords) * float(data["train_fraction"]))))
    for block in unique[:-1]:
        train |= blocks == block
        if int(train.sum()) >= target:
            break
    if train.all() or not train.any():
        raise ValueError("Invalid spatial block split")
    return train


def prepare_patches(knn, members, context_cells, stride, seed):
    """Cover every split cell using fixed patches from original local KNN.

    Initial center sampling is shuffled stride sampling; uncovered cells are
    subsequently selected as centers. No cross-split or arbitrary distant
    filler is added. The graph is explicitly truncated to each fixed patch.
    """
    n_cells = len(members)
    candidates = np.flatnonzero(members)
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(candidates)
    covered = np.zeros(n_cells, dtype=bool)
    centers, patches = [], []

    def add(center):
        local = np.asarray(knn[center], dtype=np.int64)
        local = local[(local >= 0) & (local < n_cells)]
        local = local[(local != center) & members[local]]
        # Stable deduplication preserves the distance ordering supplied by KNN.
        _, first = np.unique(local, return_index=True)
        local = local[np.sort(first)][:context_cells - 1]
        nodes = np.full(context_cells, -1, dtype=np.int32)
        nodes[0] = center
        nodes[1:len(local) + 1] = local
        covered[nodes[nodes >= 0]] = True
        centers.append(center)
        patches.append(nodes)

    for center in candidates[::stride]:
        add(int(center))
    for center in candidates:
        if not covered[center]:
            add(int(center))
    centers = np.asarray(centers, dtype=np.int32)
    nodes = np.stack(patches) if patches else np.empty((0, context_cells), dtype=np.int32)
    seen = nodes[nodes >= 0]
    reuse = np.bincount(seen, minlength=n_cells)
    if not np.array_equal(covered, members):
        raise AssertionError("Prepared patch coverage must equal its complete split")
    if not members[seen].all():
        raise AssertionError("Cross-split node in context")
    stats = {
        "eligible_cells": int(members.sum()), "center_count": len(centers),
        "unique_context_cells": int(covered.sum()), "context_coverage_fraction": 1.,
        "max_cell_context_reuse": int(reuse.max(initial=0)),
        "mean_real_context_cells": float((nodes >= 0).sum(axis=1).mean()),
        "incomplete_patches": int(((nodes >= 0).sum(axis=1) < context_cells).sum()),
        "center_selection": "seeded_stride_then_uncovered_coverage",
    }
    return centers, nodes, stats


def validate_cache(folder, signature):
    metadata_path = folder / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Incomplete cache {folder}; remove only the incomplete directory and rebuild")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("signature") != signature:
        raise ValueError(f"Cache signature mismatch in {folder}; use a new cache_root")
    cells, genes = metadata["num_cells"], metadata["num_genes"]
    specs = {
        "nuclear_log1p_float32.npy": ((cells, genes), np.dtype("float32")),
        "cyto_log1p_float32.npy": ((cells, genes), np.dtype("float32")),
        "coords.npy": ((cells, 2), np.dtype("float32")),
        "gene_ids.npy": ((genes,), np.dtype("int64")),
    }
    for split in ("train", "val"):
        count = metadata[split]["center_count"]
        specs[f"{split}_centers.npy"] = ((count,), np.dtype("int32"))
        specs[f"{split}_nodes.npy"] = ((count, signature["data_config"]["context_cells"]), np.dtype("int32"))
    for name, (shape, dtype) in specs.items():
        array = np.load(folder / name, mmap_mode="r")
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Bad cache array {folder / name}")
    return metadata


def _source_gene_names(source):
    """Read the complete measured panel without loading expression arrays."""
    with h5py.File(source, "r") as handle:
        index_name = handle["var"].attrs.get("_index", "_index")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        return [v.decode() if isinstance(v, bytes) else str(v)
                for v in handle["var"][index_name][:]]


def reuse_slice_cache(source, destination, reusable, vocab, signature):
    """Reuse immutable large arrays only after verifying the complete panel.

    The new cache gets its own IDs, vocabulary signature, and completion
    metadata. A changed ID mapping is explicitly remapped. A changed measured
    panel intersection triggers rebuilding from raw counts, never a silent
    reuse of differently ordered/subsetted expression columns.
    """
    if not reusable.exists():
        return None
    old_metadata_path = reusable / "metadata.json"
    if not old_metadata_path.exists():
        raise ValueError(f"Cannot reuse incomplete cache {reusable}")
    old_metadata = json.loads(old_metadata_path.read_text())
    old_signature = old_metadata["signature"]
    comparable = dict(old_signature, vocab_sha256=signature["vocab_sha256"])
    if comparable != signature:
        print(f"[v2-cache] REBUILD {source.stem}: source/data contract differs from reusable cache", flush=True)
        return None
    validate_cache(reusable, old_signature)
    source_names = _source_gene_names(source)
    matched_names = [name for name in source_names if name in vocab]
    previous_names = (reusable / "gene_names.txt").read_text().splitlines()
    if len(previous_names) != old_metadata["num_genes"]:
        raise ValueError(f"Gene names/count mismatch in reusable cache {reusable}")
    if previous_names != matched_names:
        print(f"[v2-cache] REBUILD {source.stem}: matched measured panel changed "
              f"({len(previous_names)} -> {len(matched_names)} genes)", flush=True)
        return None
    new_ids = np.asarray([vocab[name] for name in matched_names], dtype=np.int64)
    if not len(new_ids) or len(np.unique(new_ids)) != len(new_ids):
        raise ValueError("Matched genes need unique vocabulary IDs")
    old_ids = np.load(reusable / "gene_ids.npy")
    remapped = not np.array_equal(old_ids, new_ids)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
    linked = {}
    try:
        for name in ("nuclear_log1p_float32.npy", "cyto_log1p_float32.npy", "coords.npy",
                     "train_centers.npy", "val_centers.npy", "train_nodes.npy", "val_nodes.npy"):
            original = (reusable / name).resolve(strict=True)
            try:
                # Independent names survive cleanup of the old run while using
                # the same immutable file data on this shared filesystem.
                os.link(original, staging / name)
                linked[name] = "hardlink"
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                (staging / name).symlink_to(original)
                linked[name] = "symlink_cross_filesystem"
        np.save(staging / "gene_ids.npy", new_ids)
        (staging / "gene_names.txt").write_text("\n".join(matched_names) + "\n")
        metadata = dict(old_metadata)
        metadata.update(
            signature=signature, cache_dir=str(destination.resolve()),
            measured_panel_genes=len(source_names), vocab_coverage=len(matched_names) / len(source_names),
            excluded_unmatched_genes=[name for name in source_names if name not in vocab],
            cache_reuse={
                "source_cache": str(reusable.resolve()),
                "source_metadata_sha256": hashlib.sha256(old_metadata_path.read_bytes()).hexdigest(),
                "source_vocab_sha256": old_signature["vocab_sha256"],
                "new_vocab_sha256": signature["vocab_sha256"],
                "verification": "same source size/mtime + data contract + exact ordered measured-panel intersection",
                "gene_ids_remapped": remapped, "large_arrays": linked,
            },
        )
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))
        validate_cache(staging, signature)
        staging.rename(destination)
        print(f"[v2-cache] REUSE-LINKED {source.stem} cells={metadata['num_cells']} "
              f"genes={len(new_ids)} ids_remapped={remapped}", flush=True)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare_slice(source, old_cache, destination, vocab, vocab_sha, data, reuse_folder=None):
    source, old_cache, destination = Path(source), Path(old_cache), Path(destination)
    stat = source.stat()
    signature = {
        "schema": CACHE_SCHEMA, "source": str(source.resolve()),
        "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
        "vocab_sha256": vocab_sha, "data_config": data,
        "source_counts": ["nuclear_counts", "extranuclear_counts"],
        "target": "float32_log1p_original_integer_counts",
        "graph": "fixed_patch_truncated_original_knn255_split_isolated",
    }
    if destination.exists():
        metadata = validate_cache(destination, signature)
        print(f"[v2-cache] REUSE {source.stem} cells={metadata['num_cells']}", flush=True)
        return metadata
    if reuse_folder is not None:
        reused = reuse_slice_cache(source, destination, Path(reuse_folder), vocab, signature)
        if reused is not None:
            return reused
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
    try:
        with h5py.File(source, "r") as handle:
            index_name = handle["var"].attrs.get("_index", "_index")
            if isinstance(index_name, bytes):
                index_name = index_name.decode()
            names = [v.decode() if isinstance(v, bytes) else str(v)
                     for v in handle["var"][index_name][:]]
            selected = np.array([i for i, gene in enumerate(names) if gene in vocab], dtype=np.int64)
            genes = np.array([vocab[names[i]] for i in selected], dtype=np.int64)
            if not len(genes) or len(np.unique(genes)) != len(genes):
                raise ValueError("Matched panel must contain unique official vocabulary genes")
            unmatched = [gene for gene in names if gene not in vocab]
            coords = np.asarray(handle["obsm"]["spatial"], dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
                raise ValueError("Expected finite cell-center coordinates in micrometers")
            cells = len(coords)
            old_coords = np.load(old_cache / "coords.npy", mmap_mode="r")
            if not np.array_equal(coords, old_coords):
                raise ValueError("Historical KNN row ordering/coordinates differ from source")
            knn = np.load(old_cache / "knn_indices.npy", mmap_mode="r")
            if knn.shape[0] != cells or knn.ndim != 2:
                raise ValueError("Historical KNN cell count differs from full source")
            np.save(staging / "coords.npy", coords)
            np.save(staging / "gene_ids.npy", genes)
            (staging / "gene_names.txt").write_text("\n".join(names[i] for i in selected) + "\n")
            maps = [open_memmap(staging / name, mode="w+", dtype=np.float32,
                                shape=(cells, len(genes))) for name in
                    ("nuclear_log1p_float32.npy", "cyto_log1p_float32.npy")]
            for start in range(0, cells, 512):
                stop = min(start + 512, cells)
                raw = [read_matrix_rows(handle["layers"][name], start, stop, len(names))
                       for name in ("nuclear_counts", "extranuclear_counts")]
                if any(not np.isfinite(x).all() or (x < 0).any()
                       or (x != np.floor(x)).any() or (x > 2**24).any() for x in raw):
                    raise ValueError("Expected exactly representable nonnegative integer molecule counts")
                original = read_matrix_rows(handle["X"], start, stop, len(names))
                if not np.array_equal(raw[0] + raw[1], original):
                    raise ValueError("Nuclear plus cytoplasmic counts must equal source X")
                for target, counts in zip(maps, raw):
                    target[start:stop] = np.log1p(counts[:, selected])
                if start == 0 or stop == cells or (start // 512) % 200 == 0:
                    print(f"[v2-cache] {source.stem} raw_rows={stop}/{cells} genes={len(genes)}/{len(names)}", flush=True)
            for array in maps:
                array.flush()
            del array, maps
            train_mask = spatial_split(coords, data)
            metadata = {
                "signature": signature, "name": source.stem,
                "cache_dir": str(destination.resolve()), "num_cells": cells,
                "num_genes": len(genes), "measured_panel_genes": len(names),
                "vocab_coverage": len(genes) / len(names), "excluded_unmatched_genes": unmatched,
                "source_precision": "original_counts_to_float32_log1p",
                "epoch_definition": "one pass through all selected patch centers; all split cells occur in context",
            }
            for split, members in (("train", train_mask), ("val", ~train_mask)):
                centers, nodes, stats = prepare_patches(
                    knn, members, int(data["context_cells"]), int(data["center_stride"]),
                    stable_seed(f"{data['seed']}:{source.stem}:{split}"))
                np.save(staging / f"{split}_centers.npy", centers)
                np.save(staging / f"{split}_nodes.npy", nodes)
                metadata[split] = stats
            (staging / "metadata.json").write_text(json.dumps(metadata, indent=2))
        # Metadata is the completion marker; rename only after every array is flushed.
        staging.rename(destination)
        print(f"[v2-cache] COMPLETE {source.stem} train={metadata['train']['center_count']} val={metadata['val']['center_count']}", flush=True)
        return metadata
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reuse-cache-root", type=Path,
                        help="Verify and link compatible V2 caches, explicitly remapping vocabulary IDs")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    base = yaml.safe_load(Path(config["base_config"]).read_text())
    if int(base["data"].get("max_cells", 0)) != 0:
        raise ValueError("Full V2 preparation refuses a truncated base dataset")
    sources, old_caches = base["data"]["h5ad_paths"], base["data"]["cache_dirs"]
    if len(sources) != len(old_caches) or len(set(sources)) != len(sources):
        raise ValueError("Source/cache lists must be complete, one-to-one, and unique")
    vocab_path = vocabulary_path(config["assets_dir"])
    vocab_sha = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    vocab = json.loads(vocab_path.read_text())
    for special in ("<pad>", "<cls>"):
        if special not in vocab:
            raise ValueError(f"Missing official vocabulary token {special}")
    data = config["data"]
    if int(data["center_stride"]) < 1 or int(data["context_cells"]) < 2:
        raise ValueError("Positive stride and at least two context cells are required")
    if not 0 < float(data.get("paired_mask_ratio", .5)) < 1:
        raise ValueError("paired_mask_ratio must lie strictly between zero and one")
    cache_root = Path(config["cache_root"])
    if args.reuse_cache_root is not None and args.reuse_cache_root.resolve() == cache_root.resolve():
        raise ValueError("A vocabulary migration needs a distinct destination cache_root")
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": CACHE_SCHEMA, "vocab_sha256": vocab_sha, "data_config": data,
        "base_config": str(Path(config["base_config"]).resolve()),
        "slice_count": len(sources), "slices": [],
        "graph_contract": "K nearest available real senders within each fixed 64-cell split-isolated patch; truncated, no two-hop completeness claim",
        "target_contract": "float32 log1p original raw counts, never float16 reconstruction",
    }
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    requests = [(source, old_cache, cache_root / Path(source).stem, vocab, vocab_sha, data,
                 args.reuse_cache_root / Path(source).stem if args.reuse_cache_root is not None else None)
                for source, old_cache in zip(sources, old_caches)]
    if args.workers == 1:
        manifest["slices"] = [prepare_slice(*request) for request in requests]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_slice, *request) for request in requests]
            # Preserve source ordering regardless of preparation completion order.
            manifest["slices"] = [future.result() for future in futures]
    manifest["total_cells"] = sum(item["num_cells"] for item in manifest["slices"])
    for split in ("train", "val"):
        manifest[f"{split}_centers"] = sum(item[split]["center_count"] for item in manifest["slices"])
        manifest[f"{split}_context_cells"] = sum(item[split]["unique_context_cells"] for item in manifest["slices"])
    temporary = cache_root / f"manifest.json.partial.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(cache_root / "manifest.json")
    print(json.dumps({key: value for key, value in manifest.items() if key != "slices"}), flush=True)


if __name__ == "__main__":
    main()
