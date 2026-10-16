"""Conservative, streaming molecule-to-compartment conversion.

This module does not infer nuclei from cell centroids, expand nuclei, bin
expression, or register images. Those operations require separately validated
adapters. One output represents one physically registered slide coordinate frame.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCHEMA_VERSION = 1
COMPARTMENTS = ("nuclear", "cytoplasmic", "membrane")
XENIUM_MAP = {"1": "nuclear", "1.0": "nuclear", "true": "nuclear",
              "0": "cytoplasmic", "0.0": "cytoplasmic", "false": "cytoplasmic"}
COSMX_MAP = {"nuclear": "nuclear", "cytoplasm": "cytoplasmic",
             "cytoplasmic": "cytoplasmic", "membrane": "membrane"}


def example_manifest(platform="xenium"):
    """Return a schema example; callers must replace and verify the evidence."""
    cosmx = platform == "cosmx"
    return {
        "schema_version": 1, "dataset_id": "replace_dataset_id", "slide_id": "slide1",
        "platform": platform, "species": "Homo sapiens", "tissue": "replace_tissue",
        "provenance": {"source_url": "https://replace-with-official-source.example/data",
                       "license": "replace_with_verified_terms", "donor_id": "unknown",
                       "biological_sample_id": "replace_sample_id"},
        "cells": {
            "path": "cells.csv", "columns": {
                "cell_id": "cell_ID" if cosmx else "cell_id",
                "x": "CenterX_global_px" if cosmx else "x_centroid",
                "y": "CenterY_global_px" if cosmx else "y_centroid",
                **({"fov": "fov"} if cosmx else {})},
            **({} if cosmx else {"fov_id": "slide"}),
            "coordinates": {"unit": "pixel" if cosmx else "micron",
                            "microns_per_unit": None if cosmx else 1.0,
                            "frame": "slide_global", "registration_verified": False,
                            "evidence": "REPLACE: verify metadata coordinates share one slide frame"}},
        "molecules": {
            "path": "transcripts.csv", "record_unit": "molecule", "columns": {
                "cell_id": "CellId" if cosmx else "cell_id",
                "gene": "target" if cosmx else "feature_name",
                "compartment": "CellComp" if cosmx else "overlaps_nucleus",
                "x": "x_FOV_px" if cosmx else "x_location",
                "y": "y_FOV_px" if cosmx else "y_location",
                **({"fov": "fov"} if cosmx else {"quality": "qv"})},
            **({} if cosmx else {"fov_id": "slide"}),
            "coordinates": {"unit": "pixel" if cosmx else "micron",
                            "frame": "fov_local" if cosmx else "slide_global"}},
        "panel": {"path": "measured_genes.csv", "gene_column": "gene",
                  "controls_excluded": True, "evidence": "REPLACE: manufacturer measured gene panel"},
        "compartments": {"label_semantics_verified": False,
                         "evidence": "REPLACE: source documentation and raw label inspection",
                         "map": COSMX_MAP if cosmx else XENIUM_MAP},
        "filters": {"unassigned_cell_ids": ["0", "UNASSIGNED"] if cosmx else ["UNASSIGNED"],
                    "quality_min": None if cosmx else 20.0,
                    "control_prefixes": ["NegControl", "Negative", "Unassigned", "BLANK_"],
                    "control_genes": [],
                    "quality_evidence": "REPLACE: documented decoded-molecule threshold/filtering"},
    }


def _required_text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _table_batches(spec, columns, root, chunk_rows):
    path = _input_path(spec, root)
    columns = list(dict.fromkeys(columns))
    suffix = path.name.lower()
    if suffix.endswith(".parquet"):
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(path)
        absent = set(columns) - set(parquet.schema_arrow.names)
        if absent:
            raise ValueError(f"Missing columns in {path}: {sorted(absent)}")
        for batch in parquet.iter_batches(batch_size=chunk_rows, columns=columns):
            yield batch.to_pandas()
    elif any(suffix.endswith(end) for end in (".csv", ".csv.gz", ".tsv", ".tsv.gz")):
        separator = spec.get("separator", "\t" if ".tsv" in suffix else ",")
        with pd.read_csv(path, sep=separator, usecols=columns, dtype="string",
                         keep_default_na=False, chunksize=chunk_rows) as reader:
            yield from reader
    else:
        raise ValueError(f"Expected CSV, TSV, optionally gzip, or Parquet: {path}")


def _input_path(spec, root):
    path = Path(_required_text(spec.get("path"), "input path"))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _strings(series):
    # Preserve leading zeros from CSV IDs and fail missing keys explicitly.
    return series.astype("string").fillna("").to_numpy(dtype=str)


def _field(batch, spec, name):
    if name in spec["columns"]:
        return _strings(batch[spec["columns"][name]])
    if name == "fov" and spec.get("fov_id") is not None:
        return np.full(len(batch), str(spec["fov_id"]), dtype=object)
    raise ValueError(f"Missing {name} mapping (or explicit fov_id)")


def _validate_table_spec(spec, label, required):
    if not isinstance(spec, dict) or not isinstance(spec.get("columns"), dict):
        raise ValueError(f"{label} requires a columns mapping")
    for key in required:
        _required_text(spec["columns"].get(key), f"{label}.columns.{key}")
    if "fov" not in spec["columns"] and not str(spec.get("fov_id", "")).strip():
        raise ValueError(f"{label} needs an explicit fov column or fov_id")


def _validate_manifest(manifest):
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported unified compartment manifest schema_version")
    for key in ("dataset_id", "slide_id", "platform", "species", "tissue"):
        _required_text(manifest.get(key), key)
    provenance = manifest.get("provenance", {})
    _required_text(provenance.get("source_url"), "provenance.source_url")
    _required_text(provenance.get("license"), "provenance.license")
    _validate_table_spec(manifest.get("cells"), "cells", ("cell_id", "x", "y"))
    _validate_table_spec(manifest.get("molecules"), "molecules", ("cell_id", "gene", "compartment", "x", "y"))
    if manifest["molecules"].get("record_unit") != "molecule":
        raise ValueError("Input records must explicitly represent individual molecules, not cell/bin counts")
    for kind in ("cells", "molecules"):
        coordinates = manifest[kind].get("coordinates", {})
        if coordinates.get("unit") not in ("micron", "pixel"):
            raise ValueError(f"{kind} coordinate unit must be explicitly micron or pixel")
    coords = manifest["cells"]["coordinates"]
    if coords.get("frame") not in ("slide_global", "single_fov"):
        raise ValueError("Cell centroids require slide_global or single_fov coordinates; local transforms are not inferred")
    if coords.get("registration_verified") is not True:
        raise ValueError("Cell coordinate registration must be verified")
    _required_text(coords.get("evidence"), "cells.coordinates.evidence")
    scale = coords.get("microns_per_unit", 1.0 if coords["unit"] == "micron" else None)
    if not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Pixel coordinates require a finite positive microns_per_unit")
    if coords["unit"] == "micron" and scale != 1.0:
        raise ValueError("Micron coordinates must use scale 1.0")
    if manifest["molecules"]["coordinates"].get("frame") not in ("slide_global", "single_fov", "fov_local"):
        raise ValueError("Molecule coordinate frame must be documented")
    panel = manifest.get("panel", {})
    if panel.get("controls_excluded") is not True:
        raise ValueError("Measured gene panel must explicitly exclude control probes")
    _required_text(panel.get("evidence"), "panel.evidence")
    compartments = manifest.get("compartments", {})
    if compartments.get("label_semantics_verified") is not True:
        raise ValueError("Nuclear/cytoplasmic label semantics must be verified before conversion")
    _required_text(compartments.get("evidence"), "compartments.evidence")
    default_map = XENIUM_MAP if manifest["platform"].lower() == "xenium" else COSMX_MAP if manifest["platform"].lower() == "cosmx" else None
    mapping = compartments.get("map", default_map)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("An explicit compartment label map is required")
    normalized = {}
    for raw, target in mapping.items():
        key = str(raw).strip().lower()
        if not key or target not in COMPARTMENTS:
            raise ValueError("Compartment map targets must be nuclear, cytoplasmic, or membrane")
        if key in normalized and normalized[key] != target:
            raise ValueError(f"Conflicting compartment mapping: {raw}")
        normalized[key] = target
    if not {"nuclear", "cytoplasmic"}.issubset(normalized.values()):
        raise ValueError("Compartment mapping must distinguish nuclear and cytoplasmic labels")
    filters = manifest.get("filters", {})
    if not isinstance(filters.get("unassigned_cell_ids"), list):
        raise ValueError("Specify unassigned_cell_ids, including an explicit empty list when appropriate")
    if any(not isinstance(x, str) for x in filters["unassigned_cell_ids"]):
        raise ValueError("unassigned_cell_ids must be strings to preserve exact ID semantics")
    _required_text(filters.get("quality_evidence"), "filters.quality_evidence")
    minimum = filters.get("quality_min")
    has_quality = "quality" in manifest["molecules"]["columns"]
    if minimum is not None and (not has_quality or not isinstance(minimum, (int, float)) or not np.isfinite(minimum)):
        raise ValueError("quality_min requires a quality column and a finite numeric threshold")
    if has_quality and minimum is None:
        raise ValueError("A quality column requires an explicit quality_min")
    if manifest["platform"].lower() == "xenium" and minimum is None:
        raise ValueError("Xenium conversion requires an explicit decoded transcript quality threshold")
    for key in ("control_prefixes", "control_genes"):
        if not isinstance(filters.get(key), list) or any(not isinstance(x, str) or not x for x in filters[key]):
            raise ValueError(f"filters.{key} must be an explicit list of nonempty strings")
    return normalized, float(scale)


def _panel_genes(spec, root):
    if "genes" in spec:
        if "path" in spec or not isinstance(spec["genes"], list):
            raise ValueError("Panel requires exactly one of genes list or path")
        genes = spec["genes"]
    else:
        column = _required_text(spec.get("gene_column"), "panel.gene_column")
        pieces = [_strings(batch[column]) for batch in _table_batches(spec, [column], root, 500_000)]
        genes = np.concatenate(pieces).tolist() if pieces else []
    if not genes or any(not isinstance(g, str) or not g.strip() or g.strip() != g for g in genes):
        raise ValueError("Panel gene identifiers must be nonempty strings without surrounding whitespace")
    result = pd.Index(genes, name="gene")
    if not result.is_unique:
        raise ValueError("Measured gene panel contains duplicate identifiers")
    return result


def _load_cells(manifest, root, scale, chunk_rows):
    spec = manifest["cells"]
    pieces = list(_table_batches(spec, spec["columns"].values(), root, chunk_rows))
    if not pieces:
        raise ValueError("Empty cell metadata")
    batch = pd.concat(pieces, ignore_index=True)
    ids, fovs = _field(batch, spec, "cell_id"), _field(batch, spec, "fov")
    if any(not str(x).strip() for x in ids) or any(not str(x).strip() for x in fovs):
        raise ValueError("Missing cell or FOV identifiers in cell metadata")
    if set(ids) & set(manifest["filters"]["unassigned_cell_ids"]):
        raise ValueError("Cell metadata contains an ID declared unassigned")
    keys = pd.MultiIndex.from_arrays([fovs, ids], names=["fov", "cell_id"])
    if not keys.is_unique:
        raise ValueError("Duplicate (FOV, cell_id) keys in cell metadata")
    if spec["coordinates"]["frame"] == "single_fov" and len(set(fovs)) != 1:
        raise ValueError("single_fov coordinates cannot combine multiple FOVs")
    xy = np.column_stack([pd.to_numeric(batch[spec["columns"][axis]], errors="coerce").to_numpy(dtype=float) for axis in ("x", "y")]) * scale
    if not np.isfinite(xy).all():
        raise ValueError("Cell centroids contain missing or nonfinite coordinates")
    prefix = [manifest["dataset_id"], manifest["slide_id"]]
    names = ["/".join(quote(str(part), safe="") for part in (*prefix, fov, cid)) for fov, cid in zip(fovs, ids)]
    obs = pd.DataFrame({"dataset_id": manifest["dataset_id"], "slide_id": manifest["slide_id"],
                        "fov": fovs, "original_cell_id": ids, "platform": manifest["platform"],
                        "species": manifest["species"], "tissue": manifest["tissue"]}, index=pd.Index(names, name="cell_uid"))
    for key in ("donor_id", "biological_sample_id", "split_group"):
        if key in manifest["provenance"]:
            obs[key] = str(manifest["provenance"][key])
    return keys, obs, xy


def _aggregate(manifest, root, cell_keys, genes, mapping, chunk_rows):
    spec, filters = manifest["molecules"], manifest["filters"]
    stats = {key: 0 for key in ("rows_seen", "unassigned", "below_quality", "control", "retained", *COMPARTMENTS)}
    levels = []
    shape = (len(cell_keys), len(genes))
    for batch in _table_batches(spec, spec["columns"].values(), root, chunk_rows):
        size = len(batch)
        stats["rows_seen"] += size
        ids, fovs = _field(batch, spec, "cell_id"), _field(batch, spec, "fov")
        feature = _field(batch, spec, "gene")
        keep = np.ones(size, dtype=bool)

        def discard(mask, counter):
            count = keep & mask
            stats[counter] += int(count.sum())
            keep[count] = False

        discard(np.isin(ids, filters["unassigned_cell_ids"]) | (ids == ""), "unassigned")
        if filters.get("quality_min") is not None:
            quality = pd.to_numeric(batch[spec["columns"]["quality"]], errors="coerce").to_numpy(dtype=float)
            discard(~np.isfinite(quality) | (quality < filters["quality_min"]), "below_quality")
        controls = np.isin(feature, filters["control_genes"])
        for prefix in filters["control_prefixes"]:
            controls |= np.char.startswith(feature, prefix)
        discard(controls, "control")
        indices = cell_keys.get_indexer(pd.MultiIndex.from_arrays([fovs, ids]))
        if np.any(keep & (indices < 0)):
            examples = [(str(fovs[i]), str(ids[i])) for i in np.flatnonzero(keep & (indices < 0))[:5]]
            raise ValueError(f"Assigned molecule (FOV, cell_id) missing from cell metadata: {examples}")
        gene_indices = genes.get_indexer(feature)
        if np.any(keep & (gene_indices < 0)):
            raise ValueError(f"Non-control molecule genes absent from measured panel: {sorted(set(feature[keep & (gene_indices < 0)]))[:5]}")
        raw = _field(batch, spec, "compartment")
        labels = np.array([mapping.get(str(x).strip().lower(), "") for x in raw])
        if np.any(keep & (labels == "")):
            raise ValueError(f"Unknown compartment labels: {sorted(set(raw[keep & (labels == '')]))[:5]}")
        for axis in ("x", "y"):
            coordinates = pd.to_numeric(batch[spec["columns"][axis]], errors="coerce").to_numpy(dtype=float)
            if np.any(keep & ~np.isfinite(coordinates)):
                raise ValueError("Retained molecules have missing or nonfinite coordinates")
        if "count" in spec["columns"]:
            weights = pd.to_numeric(batch[spec["columns"]["count"]], errors="coerce").to_numpy(dtype=float)
            if np.any(keep & (weights != 1)):
                raise ValueError("Individual molecule records must have count=1; aggregated cell/bin counts are unsupported")
        additions = []
        for label in COMPARTMENTS:
            selected = keep & (labels == label)
            n = int(selected.sum())
            stats[label] += n
            additions.append(sp.coo_matrix((np.ones(n, dtype=np.int64), (indices[selected], gene_indices[selected])), shape=shape).tocsr())
        stats["retained"] += int(keep.sum())
        print(f"[unified-transcripts] rows={stats['rows_seen']} retained={stats['retained']} "
              f"nuclear={stats['nuclear']} cytoplasmic={stats['cytoplasmic']}", flush=True)
        level = 0
        while level < len(levels) and levels[level] is not None:
            additions = [a + b for a, b in zip(additions, levels[level])]
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(additions)
        else:
            levels[level] = additions
    matrices = [sp.csr_matrix(shape, dtype=np.int64) for _ in COMPARTMENTS]
    for item in levels:
        if item is not None:
            matrices = [a + b for a, b in zip(matrices, item)]
    if not stats["nuclear"] or not stats["cytoplasmic"]:
        raise ValueError("No nuclear or no cytoplasmic transcripts retained; inspect label semantics and filtering")
    if stats["rows_seen"] != stats["retained"] + stats["unassigned"] + stats["below_quality"] + stats["control"]:
        raise RuntimeError("Filtering counts do not conserve input molecules")
    if sum(int(x.sum()) for x in matrices) != stats["retained"]:
        raise RuntimeError("Compartment counts do not conserve retained molecules")
    stats["training_transcripts"] = stats["nuclear"] + stats["cytoplasmic"]
    return matrices, stats


def _file_signature(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _hash_file(path, expected=None):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(piece)
    value = digest.hexdigest()
    if expected is not None and value.lower() != str(expected).lower():
        raise ValueError(f"SHA256 mismatch for {path}")
    return value


def convert_manifest(manifest_path, output_dir, *, chunk_rows=500_000):
    """Convert one manifest into an immutable, self-describing training package.

    Returns the QC dict. CSV and Parquet molecules are streamed; cell metadata,
    panel and final sparse matrices must fit memory. Outputs are published only
    after a persisted H5AD count/shape check succeeds.
    """
    if not isinstance(chunk_rows, int) or chunk_rows <= 0:
        raise ValueError("chunk_rows must be a positive integer")
    manifest_path, destination = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    manifest_text = manifest_path.read_text()
    manifest = json.loads(manifest_text)
    mapping, scale = _validate_manifest(manifest)
    root = manifest_path.parent
    input_specs = [(kind, manifest[kind]) for kind in ("cells", "molecules")]
    if "path" in manifest["panel"]:
        input_specs.append(("panel", manifest["panel"]))
    before = {kind: _file_signature(_input_path(spec, root)) for kind, spec in input_specs}
    genes = _panel_genes(manifest["panel"], root)
    filters = manifest["filters"]
    if any(g in filters["control_genes"] or any(g.startswith(p) for p in filters["control_prefixes"]) for g in genes):
        raise ValueError("Measured gene panel includes a declared control probe")
    keys, obs, xy = _load_cells(manifest, root, scale, chunk_rows)
    (nuclear, cyto, membrane), stats = _aggregate(manifest, root, keys, genes, mapping, chunk_rows)
    fingerprints = {}
    for kind, spec in input_specs:
        path = _input_path(spec, root)
        sha256 = _hash_file(path, spec.get("sha256"))
        if before[kind] != _file_signature(path):
            raise ValueError(f"Input changed during conversion: {path}")
        fingerprints[kind] = {"path": str(path), "bytes": before[kind][0], "sha256": sha256}
    if manifest_path.read_text() != manifest_text:
        raise ValueError("Manifest changed during conversion")
    total = nuclear + cyto
    obs["nuclear_transcripts"] = np.asarray(nuclear.sum(axis=1)).ravel()
    obs["cytoplasmic_transcripts"] = np.asarray(cyto.sum(axis=1)).ravel()
    obs["membrane_transcripts"] = np.asarray(membrane.sum(axis=1)).ravel()
    obs["has_training_counts"] = np.asarray(total.sum(axis=1)).ravel() > 0
    stats.update(cells=len(keys), measured_genes=len(genes), zero_observed_measured_genes=int((np.asarray(total.sum(axis=0)).ravel() == 0).sum()),
                 zero_training_count_cells=int((~obs["has_training_counts"]).sum()), fovs=int(obs["fov"].nunique()),
                 compartment_definition="X=nuclear+cytoplasmic; membrane stored separately; unassigned excluded",
                 spatial_unit="micron", source_files=fingerprints,
                 manifest_sha256=hashlib.sha256(manifest_text.encode()).hexdigest())
    result = ad.AnnData(total, obs=obs, var=pd.DataFrame({"measured": True}, index=genes))
    result.obsm["spatial"] = xy.astype(np.float64)
    result.layers["nuclear_counts"] = nuclear
    result.layers["extranuclear_counts"] = cyto
    if "membrane" in mapping.values() or stats["membrane"]:
        result.layers["membrane_counts"] = membrane
    result.uns["unified_compartment_schema_version"] = SCHEMA_VERSION
    result.uns["compartment_source"] = {"definition": stats["compartment_definition"], "manifest_json": manifest_text,
                                         "source_files_json": json.dumps(fingerprints, sort_keys=True), "spatial_unit": "micron"}
    result.uns["compartment_qc_json"] = json.dumps(stats, sort_keys=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity_hash = hashlib.sha256(json.dumps([manifest["dataset_id"], manifest["slide_id"]]).encode()).hexdigest()[:16]
    data_filename = quote(manifest["dataset_id"], safe="") + "--" + quote(manifest["slide_id"], safe="") + "--" + identity_hash + ".h5ad"
    if len(data_filename.encode()) > 240:
        raise ValueError("Dataset/slide identifiers are too long for an unambiguous output filename")
    lock = destination.with_name(destination.name + ".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    stage = None
    try:
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        stage = Path(tempfile.mkdtemp(prefix=destination.name + ".building-", dir=destination.parent))
        result.write_h5ad(stage / data_filename, compression="gzip")
        del result
        persisted = ad.read_h5ad(stage / data_filename)
        discrepancy = persisted.X - persisted.layers["nuclear_counts"] - persisted.layers["extranuclear_counts"]
        discrepancy.eliminate_zeros()
        if discrepancy.nnz or persisted.shape != (len(keys), len(genes)) or int(persisted.X.sum()) != stats["training_transcripts"]:
            raise RuntimeError("Persisted H5AD failed count conservation validation")
        if not persisted.obs_names.is_unique or not persisted.var_names.equals(genes):
            raise RuntimeError("Persisted H5AD lost cell or measured gene identity")
        del persisted
        (stage / "qc.json").write_text(json.dumps(stats, indent=2) + "\n")
        (stage / "source_manifest.json").write_text(manifest_text)
        (stage / "READY.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "data": data_filename,
            "data_sha256": _hash_file(stage / data_filename), "status": "validated_compartment_counts",
            "training_eligibility": manifest.get("training_eligibility", "source_policy_review_required")}, indent=2) + "\n")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite {destination}")
        stage.rename(destination)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock.unlink()
    return stats
