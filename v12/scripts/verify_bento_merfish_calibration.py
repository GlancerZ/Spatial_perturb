#!/usr/bin/env python3
"""Verify Bento MERFISH XY calibration against all raw molecules (compute only).

Published H5AD identity is checked with deterministic, evenly spaced molecules
from EVERY batch. This is explicitly a sample identity check, not a claim that
every published molecule was matched. No z calibration is inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import tempfile
import zipfile

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


EVIDENCE = (
    "Bento author's ingestion uses local x/y without scaling; MERlin v0.1.6 "
    "defines local XY as pixels and global XY as microns. All raw XY records "
    "are checked against the proposed scale plus a separate offset for each FOV. "
    "Published H5AD identity is checked with evenly spaced molecule samples "
    "from every batch; batch IDs are matched to raw FOVs, never assumed equal."
)
SOURCES = [
    "https://github.com/ckmah/bento-manuscript/blob/644d0260a0dd499094dfcfe8ae6d46621ed3fe0e/notebooks/data-ingestion/Merfish_Data_Ingestion.ipynb",
    "https://github.com/ZhuangLab/MERlin/blob/1436fc434aee7abe4fd34fac80f140ef4004ab4c/merlin/analysis/globalalign.py#L21",
    "https://github.com/YeoLab/bento-tools/blob/4a74598d3de4af4ce898e319ff0b04397014ba58/bento/io/_io.py#L353",
    "https://escholarship.org/content/qt9gr475jv/qt9gr475jv.pdf",
    "https://figshare.com/articles/dataset/22558005",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")


def _copy_capped(source, destination, size, cap):
    if size < 0 or size > cap:
        raise ValueError(f"Selected member exceeds byte cap ({size} > {cap})")
    copied = 0
    while True:
        block = source.read(min(1024 * 1024, cap - copied + 1))
        if not block:
            break
        copied += len(block)
        if copied > cap:
            raise ValueError("Archive stream exceeds byte cap")
        destination.write(block)
    if copied != size:
        raise ValueError("Archive member size mismatch")
    destination.seek(0)


@contextmanager
def raw_inputs(path, *, max_zip_bytes=1024**3, max_csv_bytes=8 * 1024**3):
    """Read selected members only; nested ZIP spool uses an anonymous temp file."""
    with tempfile.TemporaryFile() as spool:
        selected = []
        codebooks = []
        with tarfile.open(path, mode="r|gz") as archive:
            for member in archive:
                base = PurePosixPath(member.name).name
                if (base.startswith("._")
                        or "__MACOSX" in PurePosixPath(member.name).parts
                        or ".ipynb_checkpoints" in PurePosixPath(member.name).parts):
                    continue
                is_barcodes = base in ("barcodes.zip", "barcodes.csv")
                is_codebook = "codebook" in base.lower() and base.lower().endswith(".csv")
                if not (is_barcodes or is_codebook):
                    continue
                _safe_name(member.name)
                if not member.isfile():
                    raise ValueError(f"Selected member is not a regular file: {member.name}")
                if is_barcodes:
                    if selected:
                        raise ValueError("Multiple candidate barcode members are ambiguous")
                    with archive.extractfile(member) as source:
                        _copy_capped(source, spool, member.size,
                                     max_zip_bytes if base.endswith(".zip") else max_csv_bytes)
                    selected.append({"name": member.name, "bytes": member.size})
                else:
                    if member.size > 1024 * 1024:
                        raise ValueError("Codebook exceeds 1 MiB cap")
                    with archive.extractfile(member) as source:
                        payload = source.read(1024 * 1024 + 1)
                    if len(payload) != member.size:
                        raise ValueError("Codebook size mismatch")
                    codebooks.append((member.name, payload))
        if len(selected) != 1:
            raise ValueError("Expected one barcodes.zip or barcodes.csv in raw archive")
        codebook = None
        if len(codebooks) > 1:
            raise ValueError("Multiple codebooks are ambiguous")
        if codebooks:
            codebook = pd.read_csv(io.BytesIO(codebooks[0][1]))
            if "name" not in codebook or codebook["name"].isna().any():
                raise ValueError("Codebook requires a nonmissing name column")
        metadata = {"barcode_member": selected[0],
                    "codebook_member": codebooks[0][0] if codebooks else None}
        if selected[0]["name"].endswith(".csv"):
            yield spool, codebook, metadata
        else:
            with zipfile.ZipFile(spool) as nested:
                candidates = [item for item in nested.infolist()
                              if not item.is_dir() and item.filename.lower().endswith(".csv")
                              and not PurePosixPath(item.filename).name.startswith("._")]
                if len(candidates) != 1:
                    raise ValueError("Nested barcodes ZIP must contain exactly one CSV")
                member = candidates[0]
                _safe_name(member.filename)
                if stat.S_ISLNK(member.external_attr >> 16) or member.file_size > max_csv_bytes:
                    raise ValueError("Unsafe or oversized nested barcode CSV")
                metadata["nested_csv"] = {"name": member.filename, "bytes": member.file_size}
                with nested.open(member) as stream:
                    yield stream, codebook, metadata


class Column:
    """Read modern and legacy AnnData columns without loading the full AnnData."""
    def __init__(self, group, key):
        self.node = group[key]
        self.categories = None
        if isinstance(self.node, h5py.Group):
            if not {"codes", "categories"}.issubset(self.node):
                raise ValueError(f"Unsupported H5AD column encoding: {self.node.name}")
            self.categories = self.node["categories"][()]
            self.node = self.node["codes"]
        elif "__categories" in group and key in group["__categories"]:
            self.categories = group["__categories"][key][()]

    def __len__(self):
        return len(self.node)

    def read(self, selection):
        values = self.node[selection]
        if self.categories is not None:
            if np.any(values < 0):
                raise ValueError(f"Missing categorical values in {self.node.name}")
            values = self.categories[values]
        if values.dtype.kind in "OSU":
            values = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)
        return values


def published_samples(path, per_batch=256, chunk_rows=250000):
    """Uniformly sample ordinal positions per batch, covering the whole table."""
    with h5py.File(path, "r") as handle:
        frame = handle["uns/points"]
        columns = {key: Column(frame, key) for key in ("x", "y", "gene", "batch")}
        size = len(columns["x"])
        if any(len(column) != size for column in columns.values()):
            raise ValueError("Published point columns differ in length")
        obs_batches = {str(v) for v in Column(handle["obs"], "batch").read(slice(None))}
        counts = Counter()
        for start in range(0, size, chunk_rows):
            values, frequencies = np.unique(columns["batch"].read(slice(start, start + chunk_rows)).astype(str),
                                             return_counts=True)
            counts.update({str(k): int(n) for k, n in zip(values, frequencies)})
        if set(counts) != obs_batches:
            raise ValueError("Published points.batch and obs.batch sets differ")
        targets = {batch: np.unique(np.linspace(0, count - 1, min(per_batch, count), dtype=np.int64))
                   for batch, count in counts.items()}
        seen = Counter()
        row_indices, batch_labels = [], []
        for start in range(0, size, chunk_rows):
            batch_values = columns["batch"].read(slice(start, start + chunk_rows)).astype(str)
            for batch in np.unique(batch_values):
                local = np.flatnonzero(batch_values == batch)
                target = targets[batch]
                chosen = target[(target >= seen[batch]) & (target < seen[batch] + len(local))] - seen[batch]
                row_indices.extend((start + local[chosen]).tolist())
                batch_labels.extend([str(batch)] * len(chosen))
                seen[batch] += len(local)
        order = np.argsort(row_indices)
        indices = np.asarray(row_indices, dtype=np.int64)[order]
        samples = {"batch": np.asarray(batch_labels, dtype=object)[order], "row_index": indices}
        for key in ("x", "y", "gene"):
            samples[key] = np.concatenate([columns[key].read(indices[start:start + 1024])
                                           for start in range(0, len(indices), 1024)])
        if not len(indices) or not np.isfinite(np.column_stack([samples["x"], samples["y"]])).all():
            raise ValueError("Published samples are empty or have nonfinite coordinates")
        return pd.DataFrame(samples), dict(counts), sorted(obs_batches)


class Moments:
    def __init__(self):
        self.n = 0
        self.mean = np.zeros(6)
        self.cov_sum = np.zeros((6, 6))
        self.minimum = np.full(6, np.inf)
        self.maximum = np.full(6, -np.inf)

    def add(self, values):
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite raw coordinate")
        n = len(values)
        mean = values.mean(axis=0)
        centered = values - mean
        total = self.n + n
        delta = mean - self.mean
        self.cov_sum += centered.T @ centered + np.outer(delta, delta) * (self.n * n / total)
        self.mean += delta * (n / total)
        self.n = total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def result(self, scale, slope_tolerance, residual_tolerance, min_rows):
        axes = {}
        passed = self.n >= min_rows
        covariance_xy = self.cov_sum[:2, :2]
        denominator = covariance_xy[0, 0] * covariance_xy[1, 1]
        identifiable = denominator > 0 and np.linalg.det(covariance_xy) / denominator > 1e-6
        passed = passed and identifiable
        for axis, local, global_, residual in (("x", 0, 2, 4), ("y", 1, 3, 5)):
            variance = self.cov_sum[local, local]
            slope = self.cov_sum[local, global_] / variance if variance > 0 else None
            offset = float(self.mean[residual])
            max_error = max(self.maximum[residual] - offset, offset - self.minimum[residual])
            axis_passed = slope is not None and abs(slope - scale) <= slope_tolerance and max_error <= residual_tolerance
            passed = passed and axis_passed
            axes[axis] = {"fitted_slope_um_per_pixel": float(slope) if slope is not None else None,
                          "fitted_intercept_um": float(self.mean[global_] - slope * self.mean[local]) if slope is not None else None,
                          "target_scale_offset_um": offset,
                          "target_scale_max_abs_residual_um": float(max_error),
                          "target_scale_rmse_um": float(np.sqrt(max(self.cov_sum[residual, residual], 0) / self.n)),
                          "local_min": float(self.minimum[local]), "local_max": float(self.maximum[local]),
                          "passed": bool(axis_passed)}
        return {"rna_rows": self.n, "xy_identifiable": bool(identifiable),
                "passed": bool(passed), "axes": axes}


def verify(raw_tar, published_h5ad, *, microns_per_pixel=0.108, expected_fov_count=70,
           samples_per_batch=256, chunk_rows=250000, slope_tolerance=1e-6,
           residual_tolerance_um=0.005, match_tolerance_pixel=0.0001, min_rows_per_fov=32):
    if any(value <= 0 for value in (microns_per_pixel, expected_fov_count, samples_per_batch,
                                    chunk_rows, slope_tolerance, residual_tolerance_um,
                                    match_tolerance_pixel, min_rows_per_fov)):
        raise ValueError("Scale, sizes, and tolerances must be positive")
    report = {"status": "failed", "microns_per_pixel": microns_per_pixel,
              "published_h5ad_sha256": sha256(published_h5ad), "raw_sha256": sha256(raw_tar),
              "published_h5ad": str(Path(published_h5ad).resolve()), "raw_tar": str(Path(raw_tar).resolve()),
              "evidence": EVIDENCE, "primary_sources": SOURCES,
              "coordinate_scope": "XY only; same scalar applied to points and cell/nucleus polygons",
              "batch_policy": "Separate local coordinate frames; no cross-batch neighbors",
              "identity_validation": "deterministic_per_batch_molecule_sample",
              "identity_rule": "All sampled molecules must jointly identify one raw FOV; individual coordinates may recur in other FOVs",
              "all_published_molecules_identity_verified": False,
              "all_published_batches_verified": False, "verified_batch_count": 0,
              "global_z": {"used_for_calibration": False, "physical_depth_verified": False},
              "tolerances": {"slope_um_per_pixel": slope_tolerance,
                             "max_xy_residual_um": residual_tolerance_um,
                             "identity_xy_distance_pixel": match_tolerance_pixel},
              "errors": []}
    samples, published_counts, batches = published_samples(published_h5ad, samples_per_batch, chunk_rows)
    report.update(published_batch_ids=batches, published_points_per_batch=published_counts,
                  identity_sample_rows=len(samples))
    trees = {}
    for gene, frame in samples.groupby("gene", sort=False):
        trees[str(gene)] = (cKDTree(frame[["x", "y"]].to_numpy(dtype=float)), frame.index.to_numpy())
    hits = [set() for _ in range(len(samples))]
    moments = defaultdict(Moments)
    raw_rows = 0
    z_equal_fov = 0
    z_observed = 0
    with raw_inputs(raw_tar) as (stream, codebook, metadata):
        report["raw_members"] = metadata
        codebook_map = codebook["name"].astype(str).to_dict() if codebook is not None else None
        for frame in pd.read_csv(stream, chunksize=chunk_rows):
            required = {"x", "y", "global_x", "global_y", "fov"}
            if not required.issubset(frame.columns):
                raise ValueError(f"Missing raw coordinate columns: {sorted(required - set(frame.columns))}")
            if "gene" not in frame:
                if codebook_map is None or "barcode_id" not in frame:
                    raise ValueError("Need raw gene column or barcode_id plus author codebook")
                barcode_ids = pd.to_numeric(frame["barcode_id"], errors="raise")
                if not np.equal(barcode_ids, np.floor(barcode_ids)).all():
                    raise ValueError("Noninteger barcode_id")
                frame["gene"] = barcode_ids.map(codebook_map)
            if frame["gene"].isna().any():
                raise ValueError("Unmapped or missing raw gene")
            numeric_fov = pd.to_numeric(frame["fov"], errors="raise").to_numpy()
            if not np.isfinite(numeric_fov).all() or not np.equal(numeric_fov, np.floor(numeric_fov)).all():
                raise ValueError("Invalid raw FOV identifier")
            frame["fov"] = numeric_fov.astype(np.int64).astype(str)
            values = frame[["x", "y", "global_x", "global_y"]].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite raw coordinates")
            residuals = values[:, 2:] - microns_per_pixel * values[:, :2]
            combined = np.column_stack([values, residuals])
            for fov, indices in frame.groupby("fov", sort=False).indices.items():
                moments[fov].add(combined[indices])
            for gene, indices in frame.groupby("gene", sort=False).indices.items():
                if str(gene) not in trees:
                    continue
                tree, sample_ids = trees[str(gene)]
                # The nearest query is only a fast rejection step. A raw point
                # must match EVERY sampled row within the unchanged tolerance,
                # including duplicate published coordinates.
                distances, _ = tree.query(values[indices, :2],
                                          distance_upper_bound=np.nextafter(match_tolerance_pixel, np.inf))
                valid = np.isfinite(distances)
                selected = indices[valid]
                if len(selected):
                    neighbors = tree.query_ball_point(values[selected, :2], r=match_tolerance_pixel)
                    fovs = frame["fov"].to_numpy()[selected]
                    for near, fov in zip(neighbors, fovs):
                        for sample in sample_ids[near]:
                            hits[int(sample)].add(str(fov))
            if "global_z" in frame:
                z = pd.to_numeric(frame["global_z"], errors="raise").to_numpy()
                z_observed += len(z)
                z_equal_fov += int(np.count_nonzero(z == numeric_fov))
            raw_rows += len(frame)
    fits = {fov: summary.result(microns_per_pixel, slope_tolerance, residual_tolerance_um, min_rows_per_fov)
            for fov, summary in sorted(moments.items())}
    report.update(raw_rna_rows=raw_rows, raw_fov_count=len(fits), fov_fits=fits,
                  expected_fov_count=expected_fov_count)
    report["global_z"].update(observed_rows=z_observed, equal_to_fov_rows=z_equal_fov,
                             interpretation="Retained only as a diagnostic; never assigned as physical depth")
    mappings, identity = {}, {}
    for batch, subset in samples.groupby("batch", sort=False):
        matched = [hits[int(index)] for index in subset.index]
        intersection = set.intersection(*matched) if matched and all(matched) else set()
        # Coordinates are local to a FOV. One integer-position RNA can recur
        # in another FOV; it is the common FOV for the entire batch sample that
        # identifies provenance. Ambiguous batch-wide intersections still fail.
        passed = len(intersection) == 1
        ambiguous = [{"published_row_index": int(samples.at[index, "row_index"]),
                      "gene": str(samples.at[index, "gene"]),
                      "x": float(samples.at[index, "x"]), "y": float(samples.at[index, "y"]),
                      "raw_fovs": sorted(hits[int(index)])}
                     for index in subset.index if len(hits[int(index)]) > 1]
        identity[str(batch)] = {"sampled_rows": len(subset), "matched_rows": sum(bool(hit) for hit in matched),
                                "candidate_raw_fovs": sorted(intersection), "passed": bool(passed),
                                "pointwise_unique_rows": sum(len(hit) == 1 for hit in matched),
                                "pointwise_ambiguous_rows": len(ambiguous),
                                "pointwise_ambiguity_examples": ambiguous[:32],
                                "published_row_indices": subset["row_index"].astype(int).tolist()}
        if passed:
            mappings[str(batch)] = next(iter(intersection))
    if len(set(mappings.values())) != len(mappings):
        report["errors"].append("Multiple published batches map to the same raw FOV")
    all_identity = len(mappings) == len(batches) and len(set(mappings.values())) == len(mappings)
    all_fits = len(fits) == expected_fov_count and set(fits) == {str(i) for i in range(expected_fov_count)} and all(fit["passed"] for fit in fits.values())
    report.update(identity_by_batch=identity, published_batch_to_raw_fov=mappings,
                  verified_batch_count=len(mappings), all_published_batches_verified=all_identity,
                  all_raw_fovs_calibrated=all_fits)
    if not all_identity:
        report["errors"].append("Not every published batch has a unique raw-FOV identity match")
    if not all_fits:
        report["errors"].append("Raw FOV coverage or XY affine calibration failed")
    if all_identity and all_fits and not report["errors"]:
        report["status"] = "passed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-tar", type=Path, required=True)
    parser.add_argument("--published-h5ad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microns-per-pixel", type=float, default=0.108)
    parser.add_argument("--expected-fov-count", type=int, default=70)
    parser.add_argument("--samples-per-batch", type=int, default=256)
    parser.add_argument("--chunk-rows", type=int, default=250000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing verification: {args.output}")
    try:
        report = verify(args.raw_tar, args.published_h5ad, microns_per_pixel=args.microns_per_pixel,
                        expected_fov_count=args.expected_fov_count, samples_per_batch=args.samples_per_batch,
                        chunk_rows=args.chunk_rows)
    except Exception as error:
        report = {"status": "failed", "errors": [f"{type(error).__name__}: {error}"],
                  "all_published_batches_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report.get(key) for key in ("status", "raw_rna_rows", "raw_fov_count",
                                                      "verified_batch_count", "errors")}, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
