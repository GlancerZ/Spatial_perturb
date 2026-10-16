#!/usr/bin/env python3
"""Build per-domain geometry and current V7 data caches without starting training."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
import yaml
from prepare_subcellular_v2 import prepare_slice, read_matrix_rows
from spatial_pretrain.data.subcellular_v2 import CACHE_SCHEMA


def string_column(element):
    if isinstance(element, h5py.Dataset):
        values = element[:]
    else:
        codes = element["codes"][:]
        if (codes < 0).any():
            raise ValueError("Coordinate domain/species cannot contain missing values")
        values = element["categories"][:][codes]
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values])


def build_geometry(coords, groups, k=255):
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Expected finite Nx2 micrometer coordinates")
    if len(groups) != len(coords) or any(not str(v).strip() for v in groups):
        raise ValueError("Every cell needs an explicit coordinate domain")
    if k < 1:
        raise ValueError("k must be positive")
    neighbors = np.full((len(coords), k), -1, dtype=np.int32)
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        take = min(k + 1, len(members))
        if take < 2:
            continue
        local = NearestNeighbors(n_neighbors=take, algorithm="kd_tree").fit(coords[members])
        indices = local.kneighbors(coords[members], return_distance=False)
        for row, adjacent in enumerate(indices):
            keep = adjacent[adjacent != row][:k]
            neighbors[members[row], :len(keep)] = members[keep]
    valid_rows, valid_cols = np.where(neighbors >= 0)
    if (neighbors[valid_rows, valid_cols] == valid_rows).any():
        raise AssertionError("Self-neighbor in geometry")
    if not np.array_equal(groups[valid_rows], groups[neighbors[valid_rows, valid_cols]]):
        raise AssertionError("Cross-domain edge in geometry")
    return neighbors


def verify_bundle(source):
    """Require the converter's immutable package, not a caller's unit assertion."""
    ready = json.loads((source.parent / "READY.json").read_text())
    if ready.get("data") != source.name or ready.get("status") != "validated_compartment_counts":
        raise ValueError("H5AD is not the validated file named by READY.json")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != ready.get("data_sha256"):
        raise ValueError("H5AD checksum differs from validated package")
    with h5py.File(source, "r") as handle:
        schema = handle["uns/unified_compartment_schema_version"][()]
        unit = handle["uns/compartment_source/spatial_unit"][()]
        if isinstance(unit, bytes):
            unit = unit.decode()
        if schema != 1 or unit != "micron":
            raise ValueError("Package must contain verified unified schema and micrometer coordinates")
        manifest = json.loads(handle["uns/compartment_source/manifest_json"][()])
    external_manifest = json.loads((source.parent / "source_manifest.json").read_text())
    if external_manifest != manifest:
        raise ValueError("Source manifest differs from the checksum-protected H5AD manifest")
    if manifest["cells"]["coordinates"].get("registration_verified") is not True:
        raise ValueError("Source coordinate registration has not been verified")
    if manifest.get("platform", "").lower() == "cosmx" and ready.get("reference_validation") != "passed":
        raise ValueError("CosMx cache requires independent vendor-expression reconstruction to pass")
    if ready.get("training_eligibility") not in {
        "vendor_expression_reconstruction_verified", "validated_nuclear_geometry_and_counts",
    }:
        raise ValueError("Package training eligibility has not passed source validation")


def geometry_fingerprint(geometry, groups, domain_column):
    fingerprint = {"domain_column": domain_column,
                   "groups_sha256": hashlib.sha256(json.dumps(groups.tolist()).encode()).hexdigest()}
    for name in ("coords.npy", "knn_indices.npy"):
        digest = hashlib.sha256()
        with (geometry / name).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024**2), b""):
                digest.update(block)
        fingerprint[name + "_sha256"] = digest.hexdigest()
    return fingerprint


def check_loader(runtime_config):
    from spatial_pretrain.data.subcellular_v7 import V7Dataset, collate_v7
    checks = {}
    for split in ("train", "val"):
        dataset = V7Dataset(runtime_config, split=split)
        indices = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
        examples = [dataset[i] for i in indices]
        batch = collate_v7(examples, pad_token_id=dataset.pad_id)
        valid = batch["query_valid_mask"]
        targets = batch["target_bins"][valid]
        if not len(targets) or (targets < 0).any() or (targets > 50).any():
            raise ValueError("V7 batch has invalid masked reconstruction targets")
        checks[split] = {"patches": len(dataset), "checked_indices": indices,
                         "target_shape": list(batch["target_bins"].shape),
                         "valid_targets": int(valid.sum())}
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Read data settings only; original training config is not changed")
    parser.add_argument("--domain-column", required=True,
                        help="Explicit obs grouping, e.g. fov; graph never crosses a group")
    parser.add_argument("--species", required=True)
    parser.add_argument("--coordinate-unit", choices=["micrometer"], required=True)
    parser.add_argument("--exclude-empty-cells", action="store_true",
                        help="Create a separate nonzero nuclear+cytoplasmic training view; preserve archival H5AD")
    args = parser.parse_args()
    verify_bundle(args.h5ad)
    runtime_config = yaml.safe_load(args.config.read_text())
    # V5/V6/V7 remove this runtime packing budget before comparing cache signatures.
    data = dict(runtime_config["data"])
    data.pop("whole_cell_max_tokens", None)
    vocab = json.loads(args.vocab.read_text())
    vocab_sha = hashlib.sha256(args.vocab.read_bytes()).hexdigest()
    with h5py.File(args.h5ad, "r") as handle:
        coords = np.asarray(handle["obsm/spatial"], dtype=np.float32)
        groups = string_column(handle["obs"][args.domain_column])
        species = string_column(handle["obs/species"])
        full_panel_nonempty = np.asarray(handle["obs/has_training_counts"], dtype=bool)
        index_name = handle["var"].attrs.get("_index", "_index")
        if isinstance(index_name, bytes):
            index_name = index_name.decode()
        genes = string_column(handle["var"][index_name])
        matched = np.flatnonzero(np.isin(genes, list(vocab)))
        if not len(matched):
            raise ValueError("No genes match the verified model vocabulary")
        keep = np.zeros(len(coords), dtype=bool)
        for start in range(0, len(coords), 512):
            stop = min(start + 512, len(coords))
            counts = read_matrix_rows(handle["X"], start, stop, len(genes))
            keep[start:stop] = (counts[:, matched] > 0).any(axis=1)
        if not keep.all() and not args.exclude_empty_cells:
            raise ValueError("Zero nuclear+cytoplasmic cells need an explicitly filtered training view; archival rows are preserved")
        if set(species) != {args.species}:
            raise ValueError("Source species differs from declared model species")
        # The present pretrained vocabulary is human; do not silently uppercase mouse genes.
        if args.species != "Homo sapiens":
            raise ValueError("Current V7 human scGPT ingestion requires Homo sapiens; preserve other species separately")
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.h5ad
    view = {"archive_h5ad": str(source.resolve()), "criterion": "nuclear+cytoplasmic total over model-matched measured genes > 0",
            "archival_cells": len(keep), "kept_cells": int(keep.sum()), "excluded_cells": int((~keep).sum()),
            "empty_full_panel_cells": int((~full_panel_nonempty).sum()),
            "empty_only_after_vocab_matching": int((full_panel_nonempty & ~keep).sum())}
    if not keep.all():
        import anndata as ad
        data_view = ad.read_h5ad(source)[keep].copy()
        source = args.output / (args.h5ad.stem + "__nonzero.h5ad")
        data_view.uns["training_view_json"] = json.dumps(view)
        data_view.write_h5ad(source, compression="gzip")
        del data_view
        coords, groups = coords[keep], groups[keep]
    if len(coords) < 2:
        raise ValueError("Not enough nonempty cells for spatial training")
    (args.output / "training_view.json").write_text(json.dumps(view, indent=2))
    geometry = args.output / "geometry"
    geometry.mkdir()
    knn = build_geometry(coords, groups)
    np.save(geometry / "coords.npy", coords)
    np.save(geometry / "knn_indices.npy", knn)
    (geometry / "metadata.json").write_text(json.dumps({
        "source": str(source.resolve()), "domain_column": args.domain_column,
        "coordinate_unit": args.coordinate_unit, "domain_count": len(np.unique(groups)),
        "k": 255, "missing_neighbors": -1, "cross_domain_edges": 0,
    }, indent=2))
    target = args.output / "v7_cache" / source.stem
    metadata = prepare_slice(source, geometry, target, vocab, vocab_sha, data)
    # Bind this new corpus to its actual neighbor graph. Legacy prepare_slice
    # reuse deliberately fails closed against this extra signature field.
    metadata["signature"]["verified_geometry"] = geometry_fingerprint(geometry, groups, args.domain_column)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    manifest = {"schema": CACHE_SCHEMA, "vocab_sha256": vocab_sha,
                "data_config": data, "slice_count": 1, "slices": [metadata],
                "graph_contract": "per_domain_knn255_then_split_isolated_fixed_patches",
                "target_contract": "float32 log1p original raw counts",
                "training_started": False, "donor_split_reviewed": False}
    (args.output / "v7_cache/manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "base_sources.yaml").write_text(yaml.safe_dump({"data": {
        "max_cells": 0, "h5ad_paths": [str(source.resolve())],
        "cache_dirs": [str(geometry.resolve())],
    }}))
    # Exercise the real current data loader, not only the cache writer.
    runtime_config["cache_root"] = str((args.output / "v7_cache").resolve())
    runtime_config["assets_dir"] = str(args.vocab.parent.resolve())
    try:
        checks = check_loader(runtime_config)
    except BaseException:
        (args.output / "v7_cache/manifest.json").rename(args.output / "v7_cache/manifest.failed_validation.json")
        raise
    (args.output / "LOADER_VALIDATED.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps({"output": str(args.output), "cells": len(coords),
                      "domains": len(np.unique(groups)), "cache": metadata}), flush=True)


if __name__ == "__main__":
    main()
