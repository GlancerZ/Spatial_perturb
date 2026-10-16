#!/usr/bin/env python3
"""Stage public molecule coordinates without inventing compartment annotations.

Run on a compute node. ZIP members are read by exact name and never extracted.
MAT v5 loading is bounded to one capped member at a time; RNA output is streamed
in bounded row batches. This is an intermediate format, NOT a training cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat


SEQFISH_NAMES = "all_gene_Names.mat"
SEQFISH_RUNS = ("RNA_locations_run_1.mat", "RNA_locations_run_2.mat")
VIZGEN_MEMBER = "202305010900_U2OS_small_set_VMSC00000/region_0/detected_transcripts.csv"
DEFAULT_CAP = 256 * 1024 * 1024
RNA_SCHEMA = pa.schema([
    ("dataset_id", pa.string()), ("sample_id", pa.string()),
    ("fov", pa.string()), ("cell_id", pa.string()), ("gene", pa.string()),
    ("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64()),
    ("coordinate_unit", pa.string()), ("z_unit", pa.string()),
    ("compartment", pa.string()), ("auxiliary_label", pa.float64()),
    ("source_member", pa.string()), ("source_record", pa.int64()),
])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member(archive: zipfile.ZipFile, name: str, cap: int) -> zipfile.ZipInfo:
    path = PurePosixPath(name)
    if cap <= 0 or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member or byte cap: {name!r}")
    matches = [entry for entry in archive.infolist() if entry.filename == name]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one member named {name!r}, found {len(matches)}")
    entry = matches[0]
    if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16):
        raise ValueError(f"Archive member must be a regular file: {name}")
    if entry.file_size > cap:
        raise ValueError(f"Member {name} exceeds byte cap: {entry.file_size} > {cap}")
    return entry


def _read_member(archive: zipfile.ZipFile, name: str, cap: int) -> bytes:
    entry = _member(archive, name, cap)
    with archive.open(entry) as stream:
        payload = stream.read(cap + 1)
    if len(payload) > cap or len(payload) != entry.file_size:
        raise ValueError(f"Member byte count mismatch or byte cap exceeded: {name}")
    return payload


class _Shards:
    def __init__(self, output: Path, rows_per_shard: int):
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.directory = output / "rna"
        self.directory.mkdir()
        self.limit = rows_per_shard
        self.buffer = {name: [] for name in RNA_SCHEMA.names}
        self.rows = 0
        self.shards: list[dict] = []

    def add(self, **row):
        if set(row) != set(self.buffer):
            raise ValueError("RNA row does not match staging schema")
        for name, value in row.items():
            self.buffer[name].append(value)
        if len(self.buffer["gene"]) >= self.limit:
            self.flush()

    def flush(self):
        count = len(self.buffer["gene"])
        if not count:
            return
        name = f"part-{len(self.shards):05d}.parquet"
        pq.write_table(pa.Table.from_pydict(self.buffer, schema=RNA_SCHEMA),
                       self.directory / name, compression="zstd")
        self.shards.append({"path": f"rna/{name}", "rows": count})
        self.rows += count
        self.buffer = {name: [] for name in RNA_SCHEMA.names}


def _start(input_zip, output, dataset_id, expected_sha256, rows_per_shard):
    path, destination = Path(input_zip), Path(output)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {destination}")
    if not dataset_id or rows_per_shard <= 0:
        raise ValueError("A nonempty dataset_id and positive rows_per_shard are required")
    digest = _sha256(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(f"Input SHA256 mismatch: {digest}")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": "subcellular-rna-staging-v1",
        "dataset_id": dataset_id,
        "status": "incomplete",
        "training_ready": False,
        "compartment_available": False,
        "compartment_values": ["unknown"],
        "input": {"path": str(path.resolve()), "bytes": path.stat().st_size,
                  "sha256": digest, "expected_sha256_verified": bool(expected_sha256)},
        "members": [], "known_blockers": [],
    }
    _save(destination, manifest)
    return destination, manifest, _Shards(destination, rows_per_shard)


def _save(output, manifest):
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _finish(output, manifest, writer):
    writer.flush()
    if writer.rows == 0:
        raise ValueError("No molecule records found; refusing to mark staging complete")
    manifest.update(status="staged_pending_annotations", rna_rows=writer.rows,
                    rna_shards=writer.shards)
    _save(output, manifest)
    return manifest


def _mat_string(value) -> str:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.ndarray) and value.dtype.kind in "US":
        value = "".join(str(item) for item in value.reshape(-1))
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid gene name in allNames: {value!r}")
    return value.strip()


def stage_seqfish(input_zip, output, *, dataset_id="seqfish_plus_nih3t3_zenodo_2669683",
                  expected_gene_count=10000, expected_sha256=None,
                  rows_per_shard=100000, max_member_bytes=DEFAULT_CAP):
    """Preserve full measured panel and MATLAB FOV/cell assignment, in pixels."""
    destination, manifest, writer = _start(
        input_zip, output, dataset_id, expected_sha256, rows_per_shard)
    manifest.update(platform="seqFISH+", species="Mus musculus", panel_complete=True,
                    measured_panel="panel.parquet", coordinate_unit="pixel",
                    coordinate_frame="local_fov", coordinate_transform="identity",
                    coordinate_index_base="unverified; original values preserved",
                    z_semantics="not available; third source column is auxiliary_label",
                    memory_policy="One capped MAT member loaded per run; bounded output row batches",
                    known_blockers=["Nucleus assignment absent; whole-cell assignment alone is insufficient",
                                    "Pixel calibration and image/RNA focal-plane alignment need verification"])
    try:
        with zipfile.ZipFile(input_zip) as archive:
            names_payload = _read_member(archive, SEQFISH_NAMES, max_member_bytes)
            names_mat = loadmat(io.BytesIO(names_payload), variable_names=["allNames"])
            if "allNames" not in names_mat:
                raise ValueError("Gene names MAT must contain allNames")
            genes = [_mat_string(value) for value in names_mat["allNames"].ravel(order="F")]
            if len(genes) != expected_gene_count or len(set(genes)) != len(genes):
                raise ValueError("Measured panel length differs from expected_gene_count or contains duplicates")
            manifest["members"].append({"name": SEQFISH_NAMES, "bytes": len(names_payload),
                                        "sha256": hashlib.sha256(names_payload).hexdigest()})
            del names_payload, names_mat
            observed = set()
            samples = []
            for run_index, member in enumerate(SEQFISH_RUNS, 1):
                payload = _read_member(archive, member, max_member_bytes)
                manifest["members"].append({"name": member, "bytes": len(payload),
                                            "sha256": hashlib.sha256(payload).hexdigest()})
                mat = loadmat(io.BytesIO(payload), variable_names=["tot"])
                del payload
                if "tot" not in mat:
                    raise ValueError(f"{member} must contain tot")
                points = mat["tot"]
                if points.ndim != 3 or points.dtype != object or points.shape[2] != len(genes):
                    raise ValueError(f"{member}: expected FOV x cell x gene MATLAB cell array")
                sample_id = f"{dataset_id}:run{run_index}"
                sample_rows, cells_with_rna = 0, 0
                for fov in range(points.shape[0]):
                    for cell in range(points.shape[1]):
                        cell_has_rna = False
                        cell_id = f"{sample_id}:fov{fov}:cell{cell}"
                        for gene_index, gene in enumerate(genes):
                            locations = np.asarray(points[fov, cell, gene_index])
                            if not locations.size:
                                continue
                            if locations.ndim != 2 or locations.shape[1] != 3:
                                raise ValueError(f"{member}: RNA records require exactly x,y,auxiliary columns")
                            if not np.issubdtype(locations.dtype, np.number) or not np.isfinite(locations).all():
                                raise ValueError(f"{member}: nonnumeric or nonfinite RNA coordinate")
                            observed.add(gene)
                            cell_has_rna = True
                            for record, (x, y, auxiliary) in enumerate(locations):
                                writer.add(dataset_id=dataset_id, sample_id=sample_id,
                                           fov=str(fov), cell_id=cell_id, gene=gene,
                                           x=float(x), y=float(y), z=None,
                                           coordinate_unit="pixel", z_unit=None,
                                           compartment="unknown", auxiliary_label=float(auxiliary),
                                           source_member=member, source_record=record)
                            sample_rows += len(locations)
                        cells_with_rna += int(cell_has_rna)
                samples.append({"sample_id": sample_id, "source_shape": list(points.shape),
                                "rna_rows": sample_rows, "cells_with_rna": cells_with_rna,
                                "indexing": "zero-based FOV/cell array indices; no all-zero slot invented as a cell"})
                del points, mat
            pq.write_table(pa.table({"gene": genes, "source_gene_index": list(range(len(genes))),
                                     "assayed": [True] * len(genes),
                                     "observed": [gene in observed for gene in genes]}),
                           destination / "panel.parquet", compression="zstd")
            manifest.update(samples=samples, measured_gene_count=len(genes),
                            observed_gene_count=len(observed),
                            provenance={"source_url": "https://zenodo.org/records/2669683",
                                        "panel_variable": "all_gene_Names.mat:allNames",
                                        "assignment": "RNA_locations_run_N.mat:tot[FOV,cell,gene]",
                                        "third_column_evidence": "https://github.com/CaiGroup/seqFISH-PLUS/blob/ee6c416006fe09262526eb364c6f66edd35adc88/scripts/Callbarcodes_ALL_channels.m"})
        return _finish(destination, manifest, writer)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(destination, manifest)
        raise


def _number(value, name):
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite {name}: {value!r}")
    return result


def stage_vizgen(input_zip, output, *, dataset_id="vizgen_u2os_segmentation_example",
                 member=VIZGEN_MEMBER, expected_sha256=None, rows_per_shard=100000,
                 max_member_bytes=DEFAULT_CAP):
    """Stream unassigned Vizgen points; observed genes are NOT a measured panel."""
    destination, manifest, writer = _start(
        input_zip, output, dataset_id, expected_sha256, rows_per_shard)
    manifest.update(platform="Vizgen/MERFISH", species="Homo sapiens", panel_complete=False, measured_panel=None,
                    observed_genes="observed_genes.parquet", coordinate_unit="micron",
                    coordinate_frame="global_xy", coordinate_transform="identity",
                    z_semantics="raw global_z; physical unit/plane-index interpretation not verified",
                    cell_assignment_available=False,
                    known_blockers=["No assigned cell IDs in source transcript table",
                                    "Nucleus/cytoplasm annotation absent; image segmentation and coordinate join required",
                                    "Complete measured panel absent; observed genes do not establish unobserved assay targets",
                                    "global_z unit requires source-specific verification"],
                    provenance={"source_url": "https://vizgen.com/data-release-program/",
                                "xy_mapping": "global_x/global_y preserved in microns; local x/y are not used"})
    try:
        observed = set()
        fovs = set()
        sample_id = f"{dataset_id}:region_0"
        with zipfile.ZipFile(input_zip) as archive:
            entry = _member(archive, member, max_member_bytes)
            manifest["members"].append({"name": member, "bytes": entry.file_size, "crc32": f"{entry.CRC:08x}"})
            with archive.open(entry) as binary, io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"global_x", "global_y", "fov", "gene"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"Missing RNA columns: {sorted(required - set(reader.fieldnames or []))}")
                if len(reader.fieldnames) != len(set(reader.fieldnames)):
                    raise ValueError("Duplicate CSV columns are ambiguous")
                for record, row in enumerate(reader):
                    if None in row or any(row[name] is None for name in required):
                        raise ValueError(f"Malformed CSV record {record}")
                    gene, fov = row["gene"].strip(), row["fov"].strip()
                    if not gene or not fov:
                        raise ValueError(f"Empty gene/FOV at CSV record {record}")
                    observed.add(gene)
                    fovs.add(fov)
                    z_text = row.get("global_z")
                    z = _number(z_text, "global_z") if z_text not in (None, "") else None
                    writer.add(dataset_id=dataset_id, sample_id=sample_id,
                               fov=fov, cell_id=None, gene=gene,
                               x=_number(row["global_x"], "global_x"),
                               y=_number(row["global_y"], "global_y"), z=z,
                               coordinate_unit="micron", z_unit="unverified_source_global_z" if z is not None else None,
                               compartment="unknown", auxiliary_label=None,
                               source_member=member, source_record=record)
        genes = sorted(observed)
        pq.write_table(pa.table({"gene": genes, "observed": [True] * len(genes),
                                 "assayed": pa.array([None] * len(genes), type=pa.bool_())}),
                       destination / "observed_genes.parquet", compression="zstd")
        manifest.update(observed_gene_count=len(genes), measured_gene_count=None,
                        samples=[{"sample_id": sample_id, "fovs": sorted(fovs)}],
                        feature_semantics="Source gene labels retained verbatim; panel/control classification unavailable")
        return _finish(destination, manifest, writer)
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        _save(destination, manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("seqfish", "vizgen"), required=True)
    parser.add_argument("--input-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--rows-per-shard", type=int, default=100000)
    parser.add_argument("--max-member-bytes", type=int, default=DEFAULT_CAP)
    parser.add_argument("--expected-gene-count", type=int, default=10000)
    args = parser.parse_args()
    kwargs = {"expected_sha256": args.expected_sha256, "rows_per_shard": args.rows_per_shard,
              "max_member_bytes": args.max_member_bytes}
    if args.dataset_id:
        kwargs["dataset_id"] = args.dataset_id
    if args.platform == "seqfish":
        result = stage_seqfish(args.input_zip, args.output,
                              expected_gene_count=args.expected_gene_count, **kwargs)
    else:
        result = stage_vizgen(args.input_zip, args.output, **kwargs)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
