"""Convert verified public NSCLC CosMx archives on a compute node.

Only the official registry's cosmx_nsclc_flat_files records are accepted. Each
sample retains source hashes, decoded RNA, measured panel and a strict comparison
against the independently supplied vendor expression matrix. No downloads occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import traceback

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spatial_pretrain.data.unified_compartments import convert_manifest


OFFICIAL_README = "https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/SMI-ReadMe.html"
ROLES = {"molecules": "tx_file.csv", "cells": "metadata_file.csv", "expression": "exprmat_file.csv", "fov_positions": "fov_positions_file.csv"}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def safe_extract(archive, destination, *, max_unpacked_bytes):
    """Extract only flat CSV/readme files, never symlinks or arbitrary paths."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    extracted, total, scanned_bytes, seen_names = [], 0, 0, set()
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name or "\x00" in member.name:
                raise ValueError(f"Unsafe archive member path: {member.name}")
            if member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                raise ValueError(f"Unsafe archive member type: {member.name}")
            if member.isdir():
                continue
            scanned_bytes += member.size
            if member.size < 0 or scanned_bytes > max_unpacked_bytes:
                raise ValueError("Archive exceeds maximum unpacked byte budget")
            name = path.name
            # Flatten vendor directories, rejecting duplicate basenames. Images,
            # serialized objects, scripts and other formats are never extracted.
            allowed = name.lower().endswith(".csv") or ("readme" in name.lower() and Path(name).suffix.lower() in (".txt", ".md", ".html"))
            if not allowed:
                continue
            if name in seen_names:
                raise ValueError(f"Duplicate flattened archive filename: {name}")
            seen_names.add(name)
            if member.size < 0 or total + member.size > max_unpacked_bytes:
                raise ValueError("Archive exceeds maximum unpacked byte budget")
            target = destination / name
            input_handle = source.extractfile(member)
            if input_handle is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            written = 0
            with input_handle, target.open("xb") as output:
                while True:
                    block = input_handle.read(min(8 * 1024 * 1024, member.size - written + 1))
                    if not block:
                        break
                    written += len(block)
                    if written > member.size:
                        raise ValueError(f"Member exceeded declared size: {member.name}")
                    output.write(block)
            if written != member.size:
                raise ValueError(f"Truncated archive member: {member.name}")
            total += written
            extracted.append({"member": member.name, "path": str(target.resolve()), "bytes": written})
    return {"bytes": total, "scanned_uncompressed_bytes": scanned_bytes, "files": extracted}


def identify_files(extraction):
    roles = {}
    for role, ending in ROLES.items():
        matches = [Path(row["path"]) for row in extraction["files"] if Path(row["path"]).name.lower().endswith(ending)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {ending}; found {len(matches)}")
        roles[role] = matches[0]
    return roles


def header(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        fields = next(csv.reader(handle), [])
    if not fields or len(fields) != len(set(fields)):
        raise ValueError(f"Missing or duplicate CSV columns: {path}")
    return fields


def source_manifest(record, files, *, microns_per_pixel, coordinate_evidence):
    expected = {
        "molecules": {"fov", "cell_ID", "x_global_px", "y_global_px", "target", "CellComp"},
        "cells": {"fov", "cell_ID", "CenterX_global_px", "CenterY_global_px", "CenterX_local_px", "CenterY_local_px"},
        "expression": {"fov", "cell_ID"},
        "fov_positions": {"fov", "x_global_px", "y_global_px"},
    }
    headers = {role: header(path) for role, path in files.items()}
    for role, required in expected.items():
        if not required.issubset(headers[role]):
            raise ValueError(f"{role} lacks required actual RNA/cell fields: {sorted(required - set(headers[role]))}")
    genes = [gene for gene in headers["expression"] if gene not in ("fov", "cell_ID") and not gene.startswith("NegPrb")]
    if not genes:
        raise ValueError("Vendor expression matrix contains no measured real genes")
    if not np.isfinite(microns_per_pixel) or microns_per_pixel <= 0 or not coordinate_evidence.strip():
        raise ValueError("An explicitly verified positive pixel scale and coordinate evidence are required")
    sample = record["sample_id"]
    registration = validate_registration(files["cells"], files["fov_positions"])
    return {
        "schema_version": 1, "dataset_id": "cosmx_nsclc_" + sample, "slide_id": sample,
        "platform": "cosmx", "species": "Homo sapiens", "tissue": "non-small-cell lung cancer",
        "training_eligibility": "vendor_expression_reconstruction_pending",
        "provenance": {
            "source_url": record["source_url"], "archive_url": record["url"],
            "archive_sha256": record["sha256"], "license": "Public research dataset; original publisher terms apply",
            "donor_id": record["donor_id"], "biological_sample_id": sample,
            "split_group": "cosmx_nsclc_" + record["donor_id"],
            "donor_group_evidence": "Vendor sample naming; Lung5 replicates grouped, Lung9 replicates grouped. Cross-study clinical identity not established.",
            "compartment_documentation": OFFICIAL_README,
            "registration_validation": registration},
        "cells": {"path": str(files["cells"].resolve()),
            "columns": {"cell_id": "cell_ID", "fov": "fov", "x": "CenterX_global_px", "y": "CenterY_global_px"},
            "coordinates": {"unit": "pixel", "microns_per_unit": microns_per_pixel, "frame": "slide_global",
                            "registration_verified": True, "evidence": coordinate_evidence}},
        "molecules": {"path": str(files["molecules"].resolve()), "record_unit": "molecule",
            "columns": {"cell_id": "cell_ID", "fov": "fov", "gene": "target", "compartment": "CellComp",
                        "x": "x_global_px", "y": "y_global_px"},
            "coordinates": {"unit": "pixel", "frame": "slide_global"}},
        "panel": {"genes": genes, "controls_excluded": True,
                  "evidence": "Actual exprMat_file.csv header; fov/cell_ID and NegPrb probes excluded; zero-observed genes retained"},
        "compartments": {"label_semantics_verified": True, "evidence": OFFICIAL_README,
                         "map": {"Nuclear": "nuclear", "Cytoplasm": "cytoplasmic", "Cytoplasmic": "cytoplasmic", "Membrane": "membrane"}},
        "filters": {"unassigned_cell_ids": ["0"], "quality_min": None,
                    "control_prefixes": ["NegPrb"], "control_genes": [],
                    "quality_evidence": "Publisher decoded transcript release; no per-molecule QV column. No Xenium QV threshold imputed. " + OFFICIAL_README},
    }


def validate_registration(cell_metadata, fov_positions):
    """Verify every vendor centroid against local coordinates + FOV origin."""
    fovs = pd.read_csv(fov_positions, usecols=["fov", "x_global_px", "y_global_px"], dtype={"fov": "string"}, keep_default_na=False)
    index = pd.Index(fovs["fov"].astype(str))
    if not index.is_unique or (index == "").any():
        raise ValueError("Duplicate or missing FOV position identifiers")
    origins = fovs[["x_global_px", "y_global_px"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(origins).all():
        raise ValueError("Nonfinite FOV origin coordinates")
    stats = {"status": "passed", "cells_checked": 0, "max_absolute_error_pixels": 0.0,
             "definition": "global centroid = local centroid + corresponding FOV origin", "fov_positions_sha256": digest_file(fov_positions)}
    fields = ["fov", "CenterX_global_px", "CenterY_global_px", "CenterX_local_px", "CenterY_local_px"]
    with pd.read_csv(cell_metadata, usecols=fields, dtype={"fov": "string"}, chunksize=50_000, keep_default_na=False) as reader:
        for batch in reader:
            indices = index.get_indexer(batch["fov"].astype(str))
            if (indices < 0).any():
                raise ValueError("Cell metadata FOV missing from FOV positions")
            global_xy = batch[["CenterX_global_px", "CenterY_global_px"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            local_xy = batch[["CenterX_local_px", "CenterY_local_px"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            error = np.abs(global_xy - local_xy - origins[indices])
            if not np.isfinite(error).all() or (error > 1e-5).any():
                raise ValueError("Cell global coordinates disagree with local coordinates + FOV origin")
            stats["cells_checked"] += len(batch)
            stats["max_absolute_error_pixels"] = max(stats["max_absolute_error_pixels"], float(error.max()) if error.size else 0.0)
    if not stats["cells_checked"]:
        raise ValueError("Empty cell metadata during coordinate validation")
    return stats


def validate_vendor_counts(bundle, expression, *, chunk_rows=1000):
    bundle = Path(bundle)
    ready = json.loads((bundle / "READY.json").read_text())
    data = ad.read_h5ad(bundle / ready["data"])
    total = data.layers["nuclear_counts"] + data.layers["extranuclear_counts"]
    if "membrane_counts" in data.layers:
        total = total + data.layers["membrane_counts"]
    keys = pd.MultiIndex.from_arrays([data.obs["fov"].astype(str), data.obs["original_cell_id"].astype(str)])
    genes = data.var_names.tolist()
    seen = np.zeros(data.n_obs, dtype=bool)
    stats = {"cells": data.n_obs, "genes": data.n_vars, "mismatched_entries": 0,
             "absolute_count_difference": 0, "reference_transcripts": 0, "unassigned_expression_rows": 0,
             "definition": "nuclear + cytoplasmic + membrane equals vendor assigned-cell expression, excluding NegPrb"}
    with pd.read_csv(expression, usecols=["fov", "cell_ID", *genes], dtype={"fov": "string", "cell_ID": "string"},
                     keep_default_na=False, chunksize=chunk_rows) as reader:
        for batch in reader:
            unassigned = batch["cell_ID"].eq("0")
            stats["unassigned_expression_rows"] += int(unassigned.sum())
            batch = batch.loc[~unassigned]
            batch_keys = pd.MultiIndex.from_arrays([batch["fov"].astype(str), batch["cell_ID"].astype(str)])
            indices = keys.get_indexer(batch_keys)
            if (indices < 0).any():
                raise ValueError("Vendor expression contains assigned cells absent from converted metadata")
            if len(set(indices)) != len(indices) or seen[indices].any():
                raise ValueError("Duplicate cell keys in vendor expression matrix")
            seen[indices] = True
            numeric = batch[genes].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
            if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric != np.floor(numeric)).any() or (numeric > np.iinfo(np.int64).max).any():
                raise ValueError("Vendor expression counts must be nonnegative finite integers")
            counts = numeric.astype(np.int64)
            difference = total[indices].toarray() - counts
            stats["mismatched_entries"] += int(np.count_nonzero(difference))
            stats["absolute_count_difference"] += int(np.abs(difference).sum())
            stats["reference_transcripts"] += int(counts.sum())
    stats["missing_reference_cells"] = int((~seen).sum())
    stats["converted_transcripts"] = int(total.sum())
    stats["expression_sha256"] = digest_file(expression)
    stats["status"] = "passed" if not stats["mismatched_entries"] and not stats["missing_reference_cells"] and stats["reference_transcripts"] == stats["converted_transcripts"] else "failed"
    return stats


def process_sample(record, archive, output, *, microns_per_pixel, coordinate_evidence,
                   max_unpacked_bytes=24 * 1024**3, chunk_rows=500_000):
    archive, output = Path(archive), Path(output)
    if not re.fullmatch(r"Lung(?:5_Rep[123]|6|9_Rep[12]|12|13)", record["sample_id"]):
        raise ValueError("Only the eight verified public NSCLC archives are supported")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing sample output: {output}")
    if archive.stat().st_size != record["size_bytes"]:
        raise ValueError("Archive byte size differs from verified registry")
    sha256 = digest_file(archive)
    if sha256 != record["sha256"]:
        raise ValueError("Archive SHA256 differs from verified registry")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "archive.json", {**record, "local_path": str(archive.resolve()), "verified_sha256": sha256})
    extraction = safe_extract(archive, output / "raw", max_unpacked_bytes=max_unpacked_bytes)
    write_json(output / "extraction.json", extraction)
    files = identify_files(extraction)
    manifest = source_manifest(record, files, microns_per_pixel=microns_per_pixel, coordinate_evidence=coordinate_evidence)
    write_json(output / "source.json", manifest)
    stats = convert_manifest(output / "source.json", output / "bundle", chunk_rows=chunk_rows)
    validation = validate_vendor_counts(output / "bundle", files["expression"])
    write_json(output / "reference_qc.json", validation)
    ready_path = output / "bundle" / "READY.json"
    ready = json.loads(ready_path.read_text())
    ready["reference_validation"] = validation["status"]
    ready["training_eligibility"] = "vendor_expression_reconstruction_verified" if validation["status"] == "passed" else "rejected_vendor_expression_mismatch"
    write_json(ready_path, ready)
    if validation["status"] != "passed":
        raise ValueError(f"Vendor expression reconstruction failed: {validation}")
    summary = {"sample_id": record["sample_id"], "status": "validated", "data": str((output / "bundle" / ready["data"]).resolve()),
               "donor_id": record["donor_id"], "cells": stats["cells"], "genes": stats["measured_genes"],
               "nuclear": stats["nuclear"], "cytoplasmic": stats["cytoplasmic"], "membrane": stats["membrane"],
               "training_transcripts": stats["training_transcripts"], "archive_sha256": sha256,
               "microns_per_pixel": microns_per_pixel, "coordinate_evidence": coordinate_evidence,
               "reference_validation": validation}
    write_json(output / "COMPLETE.json", summary)
    return summary


def run_corpus(registry_path, archive_dir, output_dir, *, samples=None, archive_template="{sample_id}.tar.gz", **kwargs):
    registry = json.loads(Path(registry_path).read_text())
    records = registry.get("cosmx_nsclc_flat_files", [])
    if not records or len({record["sample_id"] for record in records}) != len(records):
        raise ValueError("Missing or duplicate CosMx registry records")
    if samples:
        absent = set(samples) - {record["sample_id"] for record in records}
        if absent:
            raise ValueError(f"Requested samples absent from verified registry: {sorted(absent)}")
        records = [record for record in records if record["sample_id"] in samples]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "corpus_status.json"
    if summary_path.exists():
        raise FileExistsError(f"Use a new output directory; existing corpus status: {summary_path}")
    results = []
    for record in records:
        sample = record["sample_id"]
        filename = archive_template.format(sample_id=sample)
        if Path(filename).is_absolute() or ".." in Path(filename).parts:
            raise ValueError("Archive template must stay inside archive-dir")
        archive = Path(archive_dir) / filename
        try:
            result = process_sample(record, archive, output_dir / sample, **kwargs)
        except Exception as error:
            result = {"sample_id": sample, "status": "failed", "error_type": type(error).__name__, "error": str(error)}
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        results.append(result)
        write_json(summary_path, {"status": "running", "samples": results})
        print(json.dumps(result), flush=True)
    summary = {"status": "completed" if all(item["status"] == "validated" for item in results) else "completed_with_failures",
               "validated": sum(item["status"] == "validated" for item in results), "failed": sum(item["status"] != "validated" for item in results),
               "samples": results}
    write_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("resources/subcellular_data_agent/acquisition_nonxenium_20260922.json"))
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--archive-template", default="{sample_id}.tar.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample", action="append")
    parser.add_argument("--microns-per-pixel", type=float, required=True)
    parser.add_argument("--coordinate-evidence", required=True)
    parser.add_argument("--max-unpacked-bytes", type=int, default=24 * 1024**3)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    args = parser.parse_args()
    summary = run_corpus(args.registry, args.archive_dir, args.output_dir, samples=args.sample,
        archive_template=args.archive_template, microns_per_pixel=args.microns_per_pixel,
        coordinate_evidence=args.coordinate_evidence, max_unpacked_bytes=args.max_unpacked_bytes, chunk_rows=args.chunk_rows)
    return 0 if summary["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
