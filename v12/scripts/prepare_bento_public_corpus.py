#!/usr/bin/env python3
"""Stage published Bento RNA/WKT geometry; convert only with documented units.

Compute-node entrypoint. Never interprets spliced/unspliced as compartments.
Each published batch remains an independent coordinate domain. No image
registration, segmentation repair, unit inference, or donor split is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from shapely import wkt
from shapely.geometry import Point
from shapely.prepared import prep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pretrain.data.unified_compartments import convert_manifest


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def _required(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} needs explicit nonempty provenance text")
    return value


def validate_spec(spec):
    if spec.get("schema_version") != 1:
        raise ValueError("Bento spec schema_version must be 1")
    for key in ("dataset_id", "platform", "species", "tissue"):
        _required(spec.get(key), key)
    source = spec.get("source", {})
    for key in ("path", "url", "license", "expected_md5"):
        _required(source.get(key), f"source.{key}")
    checksum = source["expected_md5"]
    if len(checksum) != 32 or any(c not in "0123456789abcdef" for c in checksum.lower()):
        raise ValueError("source.expected_md5 must be the publisher's 32-digit checksum")
    panel = spec.get("panel", {})
    if panel.get("complete_original_measured_panel") is not False:
        raise ValueError("This adapter preserves a published retained gene universe, not an asserted complete original panel")
    _required(panel.get("evidence"), "panel.evidence")
    _required(spec.get("geometry_evidence"), "geometry_evidence")
    _required(spec.get("batch_evidence"), "batch_evidence")
    controls = spec.get("controls", {})
    _required(controls.get("evidence"), "controls.evidence")
    for key in ("genes", "prefixes"):
        if not isinstance(controls.get(key), list) or any(not isinstance(g, str) or not g for g in controls[key]):
            raise ValueError(f"controls.{key} must be an explicit list (empty allowed with evidence)")
    coords = spec.get("coordinates", {})
    if coords.get("unit") not in ("native", "pixel", "micron"):
        raise ValueError("coordinates.unit must be native, pixel, or micron")
    scale = coords.get("microns_per_unit")
    if scale is None:
        return None
    if (isinstance(scale, bool) or not isinstance(scale, (float, int))
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("microns_per_unit must be finite and positive when supplied")
    if coords["unit"] == "native":
        raise ValueError("Unknown native units cannot be converted by guessing a scale")
    if coords["unit"] == "micron" and scale != 1:
        raise ValueError("Micron coordinates require scale 1")
    _required(coords.get("evidence"), "coordinates.evidence")
    return float(scale)


def parse_shapes(cell_text, nuclear_text):
    """Return verified published shapes or a reason to exclude the cell."""
    geometries = []
    for name, value in (("cell", cell_text), ("nucleus", nuclear_text)):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None, None, f"missing_{name}_shape"
        value = _text(value).strip()
        if value.lower() in ("", "none", "nan", "null", "<na>"):
            return None, None, f"missing_{name}_shape"
        try:
            geometry = wkt.loads(value)
        except Exception:
            return None, None, f"unreadable_{name}_shape"
        if (geometry.is_empty or not geometry.is_valid or geometry.area <= 0
                or geometry.geom_type not in ("Polygon", "MultiPolygon")
                or not np.isfinite(np.asarray(geometry.bounds)).all()):
            return None, None, f"invalid_{name}_shape"
        geometries.append(geometry)
    cell, nucleus = geometries
    if not cell.covers(nucleus):
        return None, None, "nucleus_not_contained_in_cell"
    return cell, nucleus, "eligible"


def covers_points(geometry, xy):
    """Boundary-inclusive point membership, with a Shapely 1.x fallback."""
    try:
        from shapely import covers, points, prepare
    except ImportError:
        prepared = prep(geometry)
        return np.fromiter((prepared.covers(Point(float(x), float(y))) for x, y in xy),
                           dtype=bool, count=len(xy))
    # Preparation is idempotent in Shapely 2 and accelerates millions of RNA
    # queries against the same detailed segmentation boundary.
    prepare(geometry)
    return np.asarray(covers(geometry, points(xy)), dtype=bool)


def classify_points(cell, nucleus, xy):
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Molecule coordinates must be finite Nx2")
    in_cell = covers_points(cell, xy)
    in_nucleus = covers_points(nucleus, xy) & in_cell
    labels = np.full(len(xy), "outside_cell", dtype=object)
    labels[in_cell] = "cytoplasmic"
    labels[in_nucleus] = "nuclear"
    return labels


def source_hashes(path, expected_md5):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != expected_md5.lower():
        raise ValueError("Published H5AD MD5 mismatch")
    return {"md5": md5.hexdigest(), "sha256": sha.hexdigest(), "bytes": path.stat().st_size}


def reference_counts(data):
    result = sp.csr_matrix(data.X)
    values = result.data
    if (values.dtype.kind not in "biuf" or not np.isfinite(values).all()
            or (values < 0).any() or (values != np.floor(values)).any()
            or (values > 2**53 - 1).any()):
        raise ValueError("Published X must be raw nonnegative integer RNA counts")
    return result.astype(np.int64)


def verified_calibration(path, scale, source_sha256, batch_ids):
    """Bind a successful full-release calibration to this exact H5AD."""
    path = Path(path).resolve()
    text = path.read_text()
    report = json.loads(text)
    if report.get("status") != "passed":
        raise ValueError("Calibration report status must be passed")
    if report.get("published_h5ad_sha256") != source_sha256:
        raise ValueError("Calibration does not identify this published H5AD SHA256")
    if (report.get("all_published_batches_verified") is not True
            or report.get("verified_batch_count") != len(batch_ids)
            or not isinstance(report.get("published_batch_ids"), list)
            or set(map(str, report["published_batch_ids"])) != set(batch_ids)):
        raise ValueError("Calibration must verify every published batch")
    fitted_scale = report.get("microns_per_pixel")
    if (isinstance(fitted_scale, bool) or not isinstance(fitted_scale, (int, float))
            or not np.isfinite(fitted_scale) or fitted_scale <= 0):
        raise ValueError("Calibration must document a finite positive microns_per_pixel")
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not np.isfinite(scale) or scale <= 0
            or not np.isclose(scale, fitted_scale, rtol=1e-9, atol=0)):
        raise ValueError("Requested microns_per_pixel disagrees with calibration")
    _required(report.get("evidence"), "calibration.evidence")
    return path, text, report


def stage_bento(spec_path, output, *, chunk_rows=250_000,
                microns_per_pixel=None, calibration_evidence=None):
    if not isinstance(chunk_rows, int) or not 1 <= chunk_rows <= 1_000_000:
        raise ValueError("chunk_rows must be in [1, 1000000]")
    started = time.monotonic()
    spec_path, output = Path(spec_path).resolve(), Path(output).resolve()
    spec_text = spec_path.read_text()
    spec = json.loads(spec_text)
    scale = validate_spec(spec)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if (microns_per_pixel is None) != (calibration_evidence is None):
        raise ValueError("--microns-per-pixel and --calibration-evidence must be supplied together")
    if scale is not None and calibration_evidence is None:
        raise ValueError("Physical scale requires a passed calibration evidence JSON")
    source = Path(spec["source"]["path"])
    if not source.is_absolute():
        source = spec_path.parent / source
    source = source.resolve()
    source_stat = source.stat()
    hashes = source_hashes(source, spec["source"]["expected_md5"])
    data = ad.read_h5ad(source)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Published cell and gene identifiers must be unique")
    for name in ("cell_shape", "nucleus_shape", "batch"):
        if name not in data.obs:
            raise ValueError(f"Published H5AD lacks obs.{name}")
    points = data.uns.get("points")
    if not isinstance(points, pd.DataFrame) or not {"cell", "gene", "x", "y"}.issubset(points.columns):
        raise ValueError("Published uns.points must be a DataFrame with cell,gene,x,y")
    if data.obs["batch"].isna().any():
        raise ValueError("Published batch identifiers must not be missing")
    cells, genes = pd.Index(data.obs_names.astype(str)), pd.Index(data.var_names.astype(str))
    batches = data.obs["batch"].astype(str).to_numpy()
    if any(not value.strip() for value in batches):
        raise ValueError("Published batch identifiers must not be empty")
    calibration = None
    if calibration_evidence is not None:
        calibration = verified_calibration(calibration_evidence, microns_per_pixel,
                                           hashes["sha256"], pd.unique(batches).tolist())
        _, _, report = calibration
        spec["coordinates"] = {"unit": "pixel", "microns_per_unit": float(microns_per_pixel),
                               "evidence": report["evidence"],
                               "calibration_report_sha256": hashlib.sha256(calibration[1].encode()).hexdigest()}
        scale = validate_spec(spec)
    human_model_compatible = spec["species"].strip().lower() in ("human", "homo sapiens", "9606")
    controls = spec["controls"]
    absent_controls = set(controls["genes"]) - set(genes)
    if absent_controls:
        raise ValueError(f"Declared control genes are absent: {sorted(absent_controls)}")
    control_mask = np.asarray([gene in controls["genes"] or any(gene.startswith(prefix) for prefix in controls["prefixes"]) for gene in genes])
    if "expected_count" in controls and int(control_mask.sum()) != controls["expected_count"]:
        raise ValueError("Observed control feature count differs from documented expectation")
    panel = genes[~control_mask]
    if not len(panel):
        raise ValueError("No biological genes remain after control exclusion")
    reference = reference_counts(data)
    geometries, exclusion = [], []
    for cell_text, nuclear_text in zip(data.obs["cell_shape"], data.obs["nucleus_shape"]):
        cell, nucleus, status = parse_shapes(cell_text, nuclear_text)
        geometries.append((cell, nucleus))
        exclusion.append(status)
    eligible = np.asarray([reason == "eligible" for reason in exclusion])
    output.mkdir(parents=True, exist_ok=False)
    (output / "source_spec.json").write_text(spec_text)
    write_json(output / "resolved_spec.json", spec)
    if calibration is not None:
        (output / "calibration_evidence.json").write_text(calibration[1])
    qc = {"status": "staging", "source": str(source), "source_hashes": hashes,
          "dataset_id": spec["dataset_id"], "species": spec["species"], "platform": spec["platform"],
          "source_cells": data.n_obs, "source_points": len(points), "published_genes": data.n_vars,
          "published_gene_universe_complete_original_panel": False,
          "biological_genes": len(panel), "excluded_controls": genes[control_mask].tolist(),
          "eligible_geometry_cells": int(eligible.sum()), "excluded_geometry_cells": int((~eligible).sum()),
          "geometry_exclusion_reasons": pd.Series(exclusion)[~eligible].value_counts().to_dict(),
          "coordinate_unit": spec["coordinates"]["unit"], "microns_per_unit": scale,
          "coordinate_scale_verified": scale is not None, "training_split_ready": False,
          "human_model_compatible": human_model_compatible,
          "assignment_policy": "author cell assignment checked against cell polygon; cell/nucleus boundaries included; polygon holes respected; no geometry repairs",
          "spliced_unspliced_used": False, "count_reference": "published X compared against every supplied RNA row before filtering",
          "rows_seen": 0, "control_points": 0, "excluded_geometry_points": 0,
          "outside_cell_points": 0, "nuclear_points": 0, "cytoplasmic_points": 0, "batches": []}
    write_json(output / "status.json", qc)
    pd.DataFrame({"cell_id": cells, "batch": batches, "geometry_status": exclusion}).to_csv(output / "cell_geometry_qc.csv", index=False)
    directories = {}
    for index, batch in enumerate(pd.unique(batches)):
        selected = (batches == batch) & eligible
        if not selected.any():
            qc["batches"].append({"batch": batch, "status": "no_eligible_geometry", "cells": 0})
            continue
        folder = output / f"batch_{index:04d}"
        folder.mkdir()
        cell_rows = []
        for i in np.flatnonzero(selected):
            center = geometries[i][0].centroid
            cell_rows.append({"cell_id": cells[i], "fov": batch, "x": center.x, "y": center.y})
        pd.DataFrame(cell_rows).to_csv(folder / "cells.csv", index=False)
        pd.DataFrame({"gene": panel}).to_csv(folder / "panel.csv", index=False)
        pd.DataFrame(columns=["cell_id", "fov", "gene", "compartment", "x", "y"]).to_csv(folder / "molecules.csv", index=False)
        info = {"batch": batch, "status": "staging", "cells": len(cell_rows), "directory": str(folder),
                "nuclear_points": 0, "cytoplasmic_points": 0}
        qc["batches"].append(info)
        directories[batch] = (folder, info)
    observed = sp.csr_matrix(reference.shape, dtype=np.int64)
    try:
        for start in range(0, len(points), chunk_rows):
            frame = points.iloc[start:start + chunk_rows]
            if frame[["cell", "gene", "x", "y"]].isna().any().any():
                raise ValueError("RNA rows contain missing identities or coordinates")
            cell_index = cells.get_indexer(frame["cell"].astype(str))
            gene_index = genes.get_indexer(frame["gene"].astype(str))
            if (cell_index < 0).any() or (gene_index < 0).any():
                raise ValueError("RNA identities do not match published obs/var axes")
            xy = frame[["x", "y"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            if not np.isfinite(xy).all():
                raise ValueError("Nonfinite published RNA coordinates")
            if "batch" in frame and not np.array_equal(frame["batch"].astype(str).to_numpy(), batches[cell_index]):
                raise ValueError("RNA batch disagrees with assigned cell's published batch")
            observed += sp.coo_matrix((np.ones(len(frame), dtype=np.int64), (cell_index, gene_index)), shape=reference.shape).tocsr()
            qc["rows_seen"] += len(frame)
            is_control = control_mask[gene_index]
            qc["control_points"] += int(is_control.sum())
            good = ~is_control & eligible[cell_index]
            qc["excluded_geometry_points"] += int((~is_control & ~eligible[cell_index]).sum())
            labels = np.full(len(frame), "excluded", dtype=object)
            for i in np.unique(cell_index[good]):
                selection = good & (cell_index == i)
                labels[selection] = classify_points(*geometries[i], xy[selection])
            qc["outside_cell_points"] += int((labels == "outside_cell").sum())
            for label in ("nuclear", "cytoplasmic"):
                qc[f"{label}_points"] += int((labels == label).sum())
            retained = np.isin(labels, ["nuclear", "cytoplasmic"])
            for batch in pd.unique(batches[cell_index[retained]]):
                selected = retained & (batches[cell_index] == batch)
                folder, info = directories[batch]
                rows = pd.DataFrame({"cell_id": cells[cell_index[selected]], "fov": batch,
                                     "gene": genes[gene_index[selected]], "compartment": labels[selected],
                                     "x": xy[selected, 0], "y": xy[selected, 1]})
                rows.to_csv(folder / "molecules.csv", mode="a", header=False, index=False)
                for label in ("nuclear", "cytoplasmic"):
                    info[f"{label}_points"] += int((labels[selected] == label).sum())
            write_json(output / "status.json", qc)
            print(f"[bento] {spec['dataset_id']} RNA={qc['rows_seen']}/{len(points)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        discrepancy = observed - reference
        discrepancy.eliminate_zeros()
        qc["published_X_mismatched_entries"] = discrepancy.nnz
        qc["published_X_absolute_difference"] = int(abs(discrepancy).sum())
        if discrepancy.nnz:
            raise ValueError("RNA table counts disagree with published X; no calibrated bundles were published")
        counters = ("control_points", "excluded_geometry_points", "outside_cell_points", "nuclear_points", "cytoplasmic_points")
        if sum(qc[key] for key in counters) != len(points):
            raise AssertionError("Bento filtering failed RNA count conservation")
        current = source.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise ValueError("Source H5AD changed while staging")
        if spec_path.read_text() != spec_text:
            raise ValueError("Source spec changed while staging")
        if calibration is not None and calibration[0].read_text() != calibration[1]:
            raise ValueError("Calibration evidence changed while staging")
        for batch, (folder, info) in directories.items():
            if not info["nuclear_points"] or not info["cytoplasmic_points"]:
                info["status"] = "insufficient_compartment_support"
                continue
            if scale is None:
                info["status"] = "staged_native_coordinates_need_calibration"
                continue
            if not human_model_compatible:
                info["status"] = "staged_species_requires_matching_model"
                continue
            manifest = make_unified_manifest(spec, hashes, batch, folder, scale)
            write_json(folder / "unified_manifest.json", manifest)
            stats = convert_manifest(folder / "unified_manifest.json", folder / "bundle", chunk_rows=chunk_rows)
            ready_path = folder / "bundle" / "READY.json"
            ready = json.loads(ready_path.read_text())
            ready["reference_validation"] = "passed_published_X_before_filtering"
            ready["training_split_ready"] = False
            write_json(ready_path, ready)
            info.update(status="converted_counts_pending_split_review", retained=stats["retained"])
        converted = sum(info["status"] == "converted_counts_pending_split_review" for info in qc["batches"])
        final_status = ("staged_native_coordinates_need_calibration" if scale is None else
                        "staged_species_requires_matching_model" if not human_model_compatible else
                        "completed" if converted else "no_batches_with_usable_compartments")
        qc.update(status=final_status, converted_batches=converted,
                  elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(output / "status.json", qc)
    except Exception as error:
        qc.update(status="failed", error=f"{type(error).__name__}: {error}", elapsed_seconds=round(time.monotonic() - started, 3))
        write_json(output / "status.json", qc)
        raise
    return qc


def make_unified_manifest(spec, hashes, batch, folder, scale):
    return {
        "schema_version": 1, "dataset_id": spec["dataset_id"], "slide_id": f"published_batch_{batch}",
        "platform": spec["platform"], "species": spec["species"], "tissue": spec["tissue"],
        "training_eligibility": "validated_nuclear_geometry_and_counts",
        "provenance": {"source_url": spec["source"]["url"], "license": spec["source"]["license"],
                       "source_h5ad_sha256": hashes["sha256"], "donor_id": "unknown",
                       "biological_sample_id": spec["dataset_id"],
                       "split_group": spec["dataset_id"],
                       "published_batch": batch, "batch_evidence": spec["batch_evidence"],
                       "coordinate_calibration_sha256": spec["coordinates"]["calibration_report_sha256"],
                       "panel_is_published_retained_gene_universe": True,
                       "complete_original_measured_panel": False},
        "cells": {"path": "cells.csv", "columns": {"cell_id": "cell_id", "fov": "fov", "x": "x", "y": "y"},
                  "coordinates": {"unit": spec["coordinates"]["unit"], "microns_per_unit": scale,
                                  "frame": "single_fov", "registration_verified": True,
                                  "evidence": "Published RNA and WKT polygon coordinate frame within one batch; " + spec["coordinates"]["evidence"]}},
        "molecules": {"path": "molecules.csv", "record_unit": "molecule",
                      "columns": {"cell_id": "cell_id", "fov": "fov", "gene": "gene", "compartment": "compartment", "x": "x", "y": "y"},
                      "coordinates": {"unit": spec["coordinates"]["unit"], "frame": "single_fov"}},
        "panel": {"path": "panel.csv", "gene_column": "gene", "controls_excluded": True,
                  "complete_original_measured_panel": False, "evidence": spec["panel"]["evidence"]},
        "compartments": {"label_semantics_verified": True, "evidence": spec["geometry_evidence"],
                         "map": {"nuclear": "nuclear", "cytoplasmic": "cytoplasmic"}},
        "filters": {"unassigned_cell_ids": [], "quality_min": None, "control_prefixes": [], "control_genes": [],
                    "quality_evidence": "Published author-decoded RNA; raw QV not available. Controls and cells without valid nucleus/cell polygons removed in documented staging."},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--microns-per-pixel", type=float,
                        help="Documented physical scale; requires a matching passed calibration report")
    parser.add_argument("--calibration-evidence", type=Path,
                        help="JSON proving calibration for this source SHA256 and every published batch")
    args = parser.parse_args()
    result = stage_bento(args.spec, args.output, chunk_rows=args.chunk_rows,
                         microns_per_pixel=args.microns_per_pixel,
                         calibration_evidence=args.calibration_evidence)
    print(json.dumps(result, indent=2), flush=True)
    # Staging is a successful bounded task, not a claim of physical-coordinate readiness.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
