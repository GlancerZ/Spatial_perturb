#!/usr/bin/env python3
"""Aggregate official Visium HD 2 um counts into image-segmented compartments.

Run inside a Slurm compute-node step. This script consumes Space Ranger >= 4
feature_slice.h5 and barcode_mappings.parquet from the SAME sample/run. It does
not infer segmentation, split expression heuristically, or process FASTQs.

Schema references:
https://www.10xgenomics.com/support/software/space-ranger/4.0/analysis/outputs/space-ranger-feature-slice-h5
https://www.10xgenomics.com/support/software/space-ranger/4.0/analysis/outputs/segmented-outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
from datetime import datetime, timezone
from typing import Any


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-slice", required=True, type=Path)
    parser.add_argument("--barcode-mappings", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def require_compute_node() -> dict[str, str]:
    host = socket.gethostname().split(".")[0]
    job = os.environ.get("SLURM_JOB_ID", "")
    node = os.environ.get("SLURMD_NODENAME", "").split(".")[0]
    if not job or not node or host != node:
        raise RuntimeError(
            "Run this program in an actual Slurm compute-node step: both "
            "SLURM_JOB_ID and SLURMD_NODENAME are required, and hostname must "
            f"match SLURMD_NODENAME. Observed hostname={host!r}, job={job!r}, "
            f"node={node!r}. No input data have been loaded."
        )
    return {"hostname": host, "slurm_job_id": job, "slurmd_nodename": node}


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _schema(handle: Any, parquet: Any) -> dict[str, Any]:
    """Collect bounded structural evidence, including when validation fails."""
    out: dict[str, Any] = {
        "hdf5_root_keys": list(handle.keys()),
        "hdf5_root_attribute_names": list(handle.attrs.keys()),
        "parquet_schema": str(parquet.schema_arrow),
        "parquet_num_rows": int(parquet.metadata.num_rows),
        "parquet_num_row_groups": int(parquet.metadata.num_row_groups),
        "hdf5_groups": {},
    }
    raw_meta = handle.attrs.get("metadata_json")
    if raw_meta is not None:
        out["metadata_json_preview"] = _text(raw_meta)[:50_000]
    for name in ("features", "features/target_sets", "masks", "masks/filtered", "segmentations"):
        if name not in handle:
            continue
        group = handle[name]
        children = {}
        if hasattr(group, "keys"):
            for key in list(group.keys())[:40]:
                item = group[key]
                children[key] = {
                    "kind": "dataset" if hasattr(item, "shape") else "group",
                    "shape": list(item.shape) if hasattr(item, "shape") else None,
                    "dtype": str(item.dtype) if hasattr(item, "dtype") else None,
                }
        out["hdf5_groups"][name] = children
    if "feature_slices" in handle:
        keys = list(handle["feature_slices"].keys())
        out["feature_slice_group_count"] = len(keys)
        out["feature_slice_example_keys"] = keys[:5]
        for key in keys[:2]:
            group = handle["feature_slices"][key]
            if hasattr(group, "keys"):
                out["hdf5_groups"][f"feature_slices/{key}"] = {
                    name: {"shape": list(group[name].shape), "dtype": str(group[name].dtype)}
                    for name in group.keys() if hasattr(group[name], "shape")
                }
    return out


def _integer_array(values: Any, label: str, np: Any, *, allow_bool: bool = False) -> Any:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{label}: expected a one-dimensional array, got {array.shape}")
    if array.dtype.kind == "b" and allow_bool:
        return array.astype(np.int64)
    if array.dtype.kind not in "iu":
        raise ValueError(f"{label}: integer storage is required; observed dtype={array.dtype}")
    if array.size and (np.any(array < 0) or np.any(array > np.iinfo(np.int64).max)):
        raise ValueError(f"{label}: negative values or values exceeding int64")
    return array.astype(np.int64, copy=False)


def _coo_chunks(group: Any, nrows: int, ncols: int, batch_size: int, np: Any):
    for key in ("row", "col", "data"):
        if key not in group or not hasattr(group[key], "shape"):
            raise ValueError(f"{group.name}: missing dataset {key!r}")
        if len(group[key].shape) != 1:
            raise ValueError(f"{group.name}/{key}: expected one-dimensional dataset")
    size = group["row"].shape[0]
    if group["col"].shape[0] != size or group["data"].shape[0] != size:
        raise ValueError(f"{group.name}: unequal COO array lengths")
    for start in range(0, size, batch_size):
        stop = min(start + batch_size, size)
        row = _integer_array(group["row"][start:stop], f"{group.name}/row", np)
        col = _integer_array(group["col"][start:stop], f"{group.name}/col", np)
        data = _integer_array(group["data"][start:stop], f"{group.name}/data", np, allow_bool=True)
        if np.any(row >= nrows) or np.any(col >= ncols):
            raise ValueError(f"{group.name}: COO coordinate outside metadata grid")
        yield row * ncols + col, data


def _metadata(handle: Any, np: Any) -> tuple[dict[str, Any], int, int, float]:
    if "metadata_json" not in handle.attrs:
        raise ValueError("feature_slice.h5 has no metadata_json root attribute")
    metadata = json.loads(_text(handle.attrs["metadata_json"]))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must contain an object")
    dimensions = []
    for name in ("nrows", "ncols"):
        value = metadata.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"metadata_json/{name} must be a positive integer, got {value!r}")
        dimensions.append(value)
    nrows, ncols = dimensions
    pitch = metadata.get("spot_pitch")
    if isinstance(pitch, bool) or not isinstance(pitch, (float, int)):
        raise ValueError("metadata_json/spot_pitch must be numeric, in micrometres")
    pitch = float(pitch)
    if not np.isfinite(pitch) or not np.isclose(pitch, 2.0, rtol=0, atol=1e-6):
        raise ValueError(f"Expected native 2 um grid, observed spot_pitch={pitch!r}")
    if nrows * ncols > 100_000_000:
        raise ValueError("Grid exceeds the bounded pilot limit of 100 million squares")
    return metadata, nrows, ncols, pitch


def _features(handle: Any, np: Any) -> tuple[Any, list[str], list[str], dict[str, Any]]:
    if "features/target_sets" not in handle:
        raise ValueError("No features/target_sets: measured gene panel is unknown; refusing zero filling")
    features = handle["features"]
    for name in ("id", "name", "feature_type"):
        if name not in features or len(features[name].shape) != 1:
            raise ValueError(f"features/{name} must be a one-dimensional dataset")
    feature_ids = [_text(v) for v in features["id"][:]]
    gene_names = [_text(v) for v in features["name"][:]]
    types = [_text(v) for v in features["feature_type"][:]]
    if len(feature_ids) != len(gene_names) or len(feature_ids) != len(types):
        raise ValueError("Feature metadata arrays have unequal lengths")
    if len(set(feature_ids)) != len(feature_ids) or any(not value for value in feature_ids):
        raise ValueError("Feature IDs must be unique and nonempty")
    target_sets = features["target_sets"]
    names = list(target_sets.keys())
    if len(names) != 1:
        raise ValueError(f"Expected exactly one measured target set; found {names!r}")
    target = target_sets[names[0]]
    if not hasattr(target, "shape"):
        raise ValueError("The unique target set is not an index dataset")
    indices = _integer_array(target[:], f"features/target_sets/{names[0]}", np)
    if not indices.size or np.any(indices >= len(feature_ids)):
        raise ValueError("Target set is empty or has feature indices outside the feature table")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Target-set feature indices are duplicated")
    indices = np.sort(indices)
    if any(types[int(i)] != "Gene Expression" for i in indices):
        raise ValueError("Target set contains non-Gene Expression features; explicit handling is required")
    selected_ids = [feature_ids[int(i)] for i in indices]
    selected_names = [gene_names[int(i)] for i in indices]
    if any(not value for value in selected_names):
        raise ValueError("Measured feature has an empty gene symbol")
    return indices, selected_ids, selected_names, {
        "target_set_name": names[0],
        "n_features_in_reference": len(feature_ids),
        "n_measured_genes": len(indices),
        "n_reference_features_excluded_as_unmeasured": len(feature_ids) - len(indices),
        "n_duplicate_gene_symbols": len(selected_names) - len(set(selected_names)),
        "feature_index_order": "ascending original features table index",
        "var_index": "unique original feature IDs; original gene_name retained separately",
    }


def _mappings(parquet: Any, tissue: Any, nrows: int, ncols: int, batch_size: int, np: Any):
    names = parquet.schema_arrow.names
    canonical = {}
    for name in names:
        lowered = name.lower()
        if lowered in canonical:
            raise ValueError(f"Case-ambiguous parquet column: {name}")
        canonical[lowered] = name
    required = ("square_002um", "cell_id", "in_nucleus", "in_cell")
    missing = set(required).difference(canonical)
    if missing:
        raise ValueError(f"Missing mapping columns {sorted(missing)}; actual columns={names}")
    selected = [canonical[name] for name in required]
    seen = np.zeros(nrows * ncols, dtype=np.bool_)
    cell_at = np.full(nrows * ncols, -1, dtype=np.int32)
    nucleus_at = np.zeros(nrows * ncols, dtype=np.bool_)
    id_to_index: dict[int, int] = {}
    cell_ids: list[int] = []
    square_re = re.compile(r"s_002um_(\d+)_(\d+)(?:-1)?", re.IGNORECASE)
    cell_re = re.compile(r"cellid_(\d+)(?:-1)?", re.IGNORECASE)
    n_records = n_in_cell = n_in_nucleus = n_tissue_assigned = 0
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected, use_threads=False):
        columns = [batch.column(i).to_pylist() for i in range(len(selected))]
        for barcode, cell_id, is_nucleus, is_cell in zip(*columns):
            if type(is_nucleus) is not bool or type(is_cell) is not bool:
                raise ValueError("In_nucleus and In_cell must be non-null boolean columns")
            if is_nucleus and not is_cell:
                raise ValueError(f"Nuclear square outside its assigned cell: {barcode!r}")
            if not isinstance(barcode, str):
                raise ValueError("Square_002um must be a non-null barcode string")
            match = square_re.fullmatch(barcode)
            if match is None:
                raise ValueError(f"Unsupported 2 um barcode format: {barcode!r}")
            row, col = int(match.group(1)), int(match.group(2))
            if row >= nrows or col >= ncols:
                raise ValueError(f"Mapping barcode outside metadata grid: {barcode!r}")
            flat = row * ncols + col
            if seen[flat]:
                raise ValueError(f"Duplicated 2 um mapping barcode: {barcode!r}")
            seen[flat] = True
            n_records += 1
            if not is_cell:
                continue
            match_cell = cell_re.fullmatch(cell_id) if isinstance(cell_id, str) else None
            if match_cell is None or int(match_cell.group(1)) <= 0:
                raise ValueError(f"In_cell square has invalid Cell_id: {barcode!r}, {cell_id!r}")
            numeric_id = int(match_cell.group(1))
            if numeric_id not in id_to_index:
                if len(cell_ids) >= np.iinfo(np.int32).max:
                    raise ValueError("Too many cells for the bounded int32 mapping")
                id_to_index[numeric_id] = len(cell_ids)
                cell_ids.append(numeric_id)
            cell_at[flat] = id_to_index[numeric_id]
            nucleus_at[flat] = is_nucleus
            n_in_cell += 1
            n_in_nucleus += int(is_nucleus)
            n_tissue_assigned += int(tissue[flat])
        print(f"[mappings] rows={n_records:,} cells={len(cell_ids):,}", flush=True)
    if not cell_ids:
        raise ValueError("No valid segmented cells in barcode mappings")
    return cell_at, nucleus_at, seen, cell_ids, {
        "mapping_columns": canonical,
        "n_mapping_records": n_records,
        "n_grid_squares_missing_mapping": int(len(seen) - seen.sum()),
        "n_squares_in_cell": n_in_cell,
        "n_squares_in_nucleus": n_in_nucleus,
        "n_tissue_squares_assigned_to_cell": n_tissue_assigned,
        "n_segmented_cell_ids": len(cell_ids),
    }


def _geometry(cell_at: Any, nucleus_at: Any, tissue: Any, ncols: int, ncell: int, batch_size: int, np: Any):
    squares = np.zeros(ncell, dtype=np.int64)
    nuclear_squares = np.zeros(ncell, dtype=np.int64)
    tissue_squares = np.zeros(ncell, dtype=np.int64)
    row_sum = np.zeros(ncell, dtype=np.int64)
    col_sum = np.zeros(ncell, dtype=np.int64)
    nuclear_row_sum = np.zeros(ncell, dtype=np.int64)
    nuclear_col_sum = np.zeros(ncell, dtype=np.int64)
    for start in range(0, len(cell_at), batch_size):
        stop = min(start + batch_size, len(cell_at))
        flats = np.arange(start, stop, dtype=np.int64)
        valid = cell_at[start:stop] >= 0
        flats = flats[valid]
        ids = cell_at[flats]
        np.add.at(squares, ids, 1)
        np.add.at(row_sum, ids, flats // ncols)
        np.add.at(col_sum, ids, flats % ncols)
        np.add.at(nuclear_squares, ids[nucleus_at[flats]], 1)
        np.add.at(nuclear_row_sum, ids[nucleus_at[flats]], flats[nucleus_at[flats]] // ncols)
        np.add.at(nuclear_col_sum, ids[nucleus_at[flats]], flats[nucleus_at[flats]] % ncols)
        np.add.at(tissue_squares, ids[tissue[flats]], 1)
    if np.any(squares <= 0):
        raise ValueError("A mapped cell has no geometric support")
    return squares, nuclear_squares, tissue_squares, row_sum, col_sum, nuclear_row_sum, nuclear_col_sum


def aggregate(args: argparse.Namespace, execution: dict[str, str], output_dir: Path) -> dict[str, Any]:
    # These imports are deliberately after the compute-node guard.
    import anndata as ad
    import h5py
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from scipy import sparse

    sources = {
        "feature_slice": {"path": str(args.feature_slice.resolve()), "size_bytes": args.feature_slice.stat().st_size},
        "barcode_mappings": {"path": str(args.barcode_mappings.resolve()), "size_bytes": args.barcode_mappings.stat().st_size},
    }
    parquet = pq.ParquetFile(args.barcode_mappings)
    with h5py.File(args.feature_slice, "r") as handle:
        dump_json(output_dir / "schema.json", _schema(handle, parquet))
        metadata, nrows, ncols, pitch = _metadata(handle, np)
        indices, feature_ids, gene_names, feature_audit = _features(handle, np)
        if "masks/filtered" not in handle:
            raise ValueError("Missing masks/filtered: raw-resolution tissue support is unknown")
        tissue = np.zeros(nrows * ncols, dtype=np.bool_)
        for flats, values in _coo_chunks(handle["masks/filtered"], nrows, ncols, args.batch_size, np):
            if np.any((values != 0) & (values != 1)):
                raise ValueError("masks/filtered is not a binary tissue mask")
            tissue[flats[values == 1]] = True
        if not tissue.any():
            raise ValueError("The filtered tissue mask is empty")
        cell_at, nucleus_at, seen, cell_ids, mapping_audit = _mappings(
            parquet, tissue, nrows, ncols, args.batch_size, np
        )
        ncell = len(cell_ids)
        squares, nuclear_squares, tissue_squares, row_sum, col_sum, nuclear_row_sum, nuclear_col_sum = _geometry(
            cell_at, nucleus_at, tissue, ncols, ncell, args.batch_size, np
        )
        n_data, n_rows, c_data, c_rows = [], [], [], []
        n_indptr, c_indptr = [0], [0]
        umi = {key: 0 for key in (
            "all_measured_gene_umi", "outside_tissue_umi", "tissue_unassigned_umi",
            "tissue_assigned_umi", "nuclear_umi", "cytoplasmic_umi",
        )}
        if "feature_slices" not in handle:
            raise ValueError("Missing feature_slices group")
        groups = handle["feature_slices"]
        missing_zero_genes = 0
        for gene_idx, feature_index in enumerate(indices):
            key = str(int(feature_index))
            nuclear = np.zeros(ncell, dtype=np.int64)
            cytoplasmic = np.zeros(ncell, dtype=np.int64)
            if key not in groups:
                # Official schema: measured genes with no UMIs have no slice.
                missing_zero_genes += 1
            else:
                for flats, values in _coo_chunks(groups[key], nrows, ncols, args.batch_size, np):
                    # Exact int64 sums; forbid chunks that could overflow accumulation.
                    chunk_total = sum(int(value) for value in values)
                    if umi["all_measured_gene_umi"] + chunk_total > np.iinfo(np.int64).max:
                        raise ValueError("UMI total exceeds exact int64 aggregation capacity")
                    umi["all_measured_gene_umi"] += chunk_total
                    in_tissue = tissue[flats]
                    if np.any(in_tissue & (values > 0) & ~seen[flats]):
                        raise ValueError("Positive measured-gene count under tissue lacks a barcode mapping")
                    umi["outside_tissue_umi"] += int(values[~in_tissue].sum())
                    mapped = cell_at[flats]
                    assigned = in_tissue & (mapped >= 0)
                    unassigned = in_tissue & (mapped < 0)
                    umi["tissue_unassigned_umi"] += int(values[unassigned].sum())
                    umi["tissue_assigned_umi"] += int(values[assigned].sum())
                    in_nucleus = assigned & nucleus_at[flats]
                    in_cytoplasm = assigned & ~nucleus_at[flats]
                    np.add.at(nuclear, mapped[in_nucleus], values[in_nucleus])
                    np.add.at(cytoplasmic, mapped[in_cytoplasm], values[in_cytoplasm])
                    umi["nuclear_umi"] += int(values[in_nucleus].sum())
                    umi["cytoplasmic_umi"] += int(values[in_cytoplasm].sum())
            nz_n = np.flatnonzero(nuclear).astype(np.int32)
            nz_c = np.flatnonzero(cytoplasmic).astype(np.int32)
            n_rows.append(nz_n)
            n_data.append(nuclear[nz_n])
            c_rows.append(nz_c)
            c_data.append(cytoplasmic[nz_c])
            n_indptr.append(n_indptr[-1] + len(nz_n))
            c_indptr.append(c_indptr[-1] + len(nz_c))
            if gene_idx % 500 == 0 or gene_idx == len(indices) - 1:
                print(f"[counts] measured_genes={gene_idx + 1:,}/{len(indices):,}", flush=True)
        nuclear = sparse.csc_matrix(
            (np.concatenate(n_data), np.concatenate(n_rows), np.asarray(n_indptr, dtype=np.int64)),
            shape=(ncell, len(indices)), dtype=np.int64,
        ).tocsr()
        cytoplasmic = sparse.csc_matrix(
            (np.concatenate(c_data), np.concatenate(c_rows), np.asarray(c_indptr, dtype=np.int64)),
            shape=(ncell, len(indices)), dtype=np.int64,
        ).tocsr()
        del n_data, n_rows, c_data, c_rows
        totals = nuclear + cytoplasmic
        nuclear_sum = np.asarray(nuclear.sum(axis=1)).ravel()
        cyto_sum = np.asarray(cytoplasmic.sum(axis=1)).ravel()
        total_sum = nuclear_sum + cyto_sum
        supported = tissue_squares > 0
        keep = supported & (total_sum > 0)
        if not keep.any():
            raise ValueError("No cells have both tissue support and measured RNA")
        if umi["all_measured_gene_umi"] != (
            umi["outside_tissue_umi"] + umi["tissue_unassigned_umi"] + umi["tissue_assigned_umi"]
        ):
            raise ValueError("Count conservation failed: all != outside + unassigned + assigned")
        if umi["tissue_assigned_umi"] != umi["nuclear_umi"] + umi["cytoplasmic_umi"]:
            raise ValueError("Count conservation failed: assigned != nuclear + cytoplasmic")
        if int(total_sum.sum()) != umi["tissue_assigned_umi"]:
            raise ValueError("Sparse aggregation changed the assigned UMI total")
        order = np.flatnonzero(keep)
        order = order[np.argsort(np.asarray(cell_ids, dtype=np.int64)[order])]
        totals = totals[order].copy()
        nuclear = nuclear[order].copy()
        cytoplasmic = cytoplasmic[order].copy()
        if (totals != nuclear + cytoplasmic).nnz:
            raise ValueError("Output compartment identity X = N + C failed")
        if int(totals.sum()) != umi["tissue_assigned_umi"]:
            raise ValueError("Cell filtering discarded assigned measured UMI")
        obs = pd.DataFrame(index=pd.Index(
            [f"cellid_{cell_ids[int(i)]:09d}-1" for i in order], name="cell_id"
        ))
        obs["source_cell_numeric_id"] = np.asarray(cell_ids, dtype=np.int64)[order]
        obs["n_counts"] = total_sum[order]
        obs["nuclear_counts"] = nuclear_sum[order]
        obs["cytoplasmic_counts"] = cyto_sum[order]
        obs["n_genes"] = totals.getnnz(axis=1)
        obs["nuclear_n_genes"] = nuclear.getnnz(axis=1)
        obs["cytoplasmic_n_genes"] = cytoplasmic.getnnz(axis=1)
        obs["has_nuclear_rna"] = nuclear_sum[order] > 0
        obs["has_cytoplasmic_rna"] = cyto_sum[order] > 0
        obs["n_assigned_squares"] = squares[order]
        obs["n_nuclear_squares"] = nuclear_squares[order]
        obs["n_tissue_squares"] = tissue_squares[order]
        obs["area_um2"] = squares[order] * pitch**2
        obs["nuclear_area_um2"] = nuclear_squares[order] * pitch**2
        obs["sample_id"] = _text(metadata.get("sample_id", "unknown"))
        obs["tech"] = "visium_hd"
        obs["observation_type"] = "image_segmented_cell"
        var = pd.DataFrame({
            "gene_name": gene_names, "feature_id": feature_ids,
            "source_feature_index": indices,
            "measured": np.ones(len(indices), dtype=bool),
        }, index=pd.Index(feature_ids, name="feature_id"))
        # A zero-based square centre is represented by (column, row) * pitch.
        # The origin is arbitrary; a global half-pitch offset changes no distances.
        spatial_um = np.column_stack((col_sum / squares, row_sum / squares))[order] * pitch
        nuclear_spatial_um = np.full((len(order), 2), np.nan, dtype=np.float32)
        nuclear_centered = nuclear_squares[order] > 0
        nuclear_order = order[nuclear_centered]
        nuclear_spatial_um[nuclear_centered] = np.column_stack((
            nuclear_col_sum[nuclear_order] / nuclear_squares[nuclear_order],
            nuclear_row_sum[nuclear_order] / nuclear_squares[nuclear_order],
        )) * pitch
        for name, source in sources.items():
            source["sha256"] = sha256(Path(source["path"]))
        provenance = {
            "method": "official_space_ranger_image_segmentations_and_2um_square_assignment",
            "nuclear_definition": "In_cell AND In_nucleus AND tissue_mask",
            "cytoplasmic_definition": "In_cell AND NOT In_nucleus AND tissue_mask",
            "count_source": "raw integer UMI in feature_slices; unique measured target_set only",
            "coordinates": "unweighted geometric centroid of ALL assigned 2um square centres; array axes",
            "coordinate_units": "micrometres",
            "coordinate_origin": "array square(0,0) centre; x=column*spot_pitch, y=row*spot_pitch",
            "centroid_is_official_polygon_centroid": False,
            "rna_level_compartment_localization": False,
            "limitation": "Image/bin-derived compartment estimates; not transcript-resolved localization.",
            "sources": sources,
            "execution": execution,
            "target_set_name": feature_audit["target_set_name"],
            "sample_id": _text(metadata.get("sample_id", "unknown")),
            "nrows": nrows, "ncols": ncols, "spot_pitch_um": pitch,
        }
        adata = ad.AnnData(X=totals, obs=obs, var=var)
        adata.layers["nuclear_counts"] = nuclear
        adata.layers["cytoplasmic_counts"] = cytoplasmic
        adata.obsm["spatial_um"] = spatial_um.astype(np.float32)
        adata.obsm["nuclear_spatial_um"] = nuclear_spatial_um
        adata.uns["compartment_provenance"] = provenance
        adata.uns["input_schema"] = "visium_hd_segmented_compartments_v1"
        output = output_dir / "compartments.h5ad"
        temporary = output_dir / "compartments.partial.h5ad"
        adata.write_h5ad(temporary, compression="gzip")
        temporary.replace(output)
        result = {
            "status": "complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_h5ad": str(output.resolve()), "execution": execution,
            "sources": sources, "features": feature_audit, "mappings": mapping_audit,
            "sample_id": provenance["sample_id"], "grid": {"nrows": nrows, "ncols": ncols, "spot_pitch_um": pitch},
            "n_tissue_squares": int(tissue.sum()),
            "n_measured_genes_without_observed_umi_slice": missing_zero_genes,
            "n_cells_before_filter": ncell,
            "n_cells_without_tissue_support": int((~supported).sum()),
            "n_cells_with_tissue_but_no_rna": int((supported & (total_sum == 0)).sum()),
            "n_cells_retained": len(order),
            "fraction_retained_missing_nuclear_rna": float((nuclear_sum[order] == 0).mean()),
            "fraction_retained_missing_cytoplasmic_rna": float((cyto_sum[order] == 0).mean()),
            "umi": umi,
            "checks": {"partition_conservation": True, "sparse_aggregation_conservation": True,
                       "cell_filter_conservation": True, "X_equals_nuclear_plus_cytoplasmic": True},
            "coordinate_provenance": provenance["coordinates"],
            "inference_or_biological_validation_performed": False,
        }
        dump_json(output_dir / "source_sha256.json", sources)
        return result


def main() -> int:
    args = args_parser().parse_args()
    execution = require_compute_node()
    if args.batch_size < 1 or args.batch_size > 1_000_000:
        raise ValueError("--batch-size must be between 1 and 1000000")
    if not args.feature_slice.is_file() or not args.barcode_mappings.is_file():
        raise FileNotFoundError("Both --feature-slice and --barcode-mappings must exist")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = aggregate(args, execution, output_dir)
    except Exception as exc:
        dump_json(output_dir / "audit.json", {
            "status": "failed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "execution": execution, "error_type": type(exc).__name__, "error": str(exc),
            "schema_path": str(output_dir / "schema.json"),
            "note": "Partial outputs, if present, are not validated and must not be used.",
        })
        raise
    dump_json(output_dir / "audit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
