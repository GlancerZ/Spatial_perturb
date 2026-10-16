"""Process one Xenium outs.zip into paired total/nuclear/cytoplasmic matrices.

Run only on a verified Slurm compute node with the scviva-tools environment.
``--initialize`` only creates the monitoring manifest and pending states. The
input H5AD is built anew from the archive's matrix and cells table; existing
processed H5ADs in the source manifest are never used or changed.

ResolVI corrects cell-by-gene total counts. Nuclear and extranuclear corrected
values below are *proportional estimates* based on the original split. ResolVI
does not reassign molecules, change cell IDs, or infer a new compartment.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import anndata as ad
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from spatial_pretrain.data.xenium_compartments import convert_xenium_compartments
from standardize_spatial_h5ad import convert_xenium_zip_file

DEFAULT_MANIFEST = ROOT / "data/pretrain_data/manifests/xenium49_compartment_inputs.tsv"
DEFAULT_OUTPUT = ROOT / "outputs/xenium_raw_resolvi_20260927"
SAMPLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
POSTERIOR_PATTERN = re.compile(r"\[resolvi\] posterior cells=(\d+)/(\d+)")
TRANSCRIPT_PATTERN = re.compile(r"\[transcripts\] rows=(\d+) retained=(\d+)")
EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)/(\d+):")


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def ensure_pending(path: Path, data: dict) -> None:
    """Install a pending state without replacing a concurrent worker's state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or not {"sample", "archive"}.issubset(rows[0]):
        raise ValueError(f"Manifest lacks sample/archive rows: {path}")
    seen: set[str] = set()
    for row in rows:
        sample = row["sample"]
        if not SAMPLE_PATTERN.fullmatch(sample) or sample in seen:
            raise ValueError(f"Invalid or repeated sample_id: {sample!r}")
        seen.add(sample)
    return rows


def archive_path(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else ROOT / path


def initialize(rows: list[dict[str, str]], output_root: Path, source_manifest: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    entries = [{"sample_id": row["sample"], "name": row["sample"]} for row in rows]
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("samples") != entries:
            raise ValueError(f"Existing monitoring manifest differs: {manifest_path}")
    else:
        atomic_json(manifest_path, {"samples": entries, "created_at": timestamp(),
                                    "source_manifest": str(source_manifest)})
    for row in rows:
        sample = row["sample"]
        ensure_pending(output_root / "state" / f"{sample}.json", {
            "sample_id": sample, "status": "pending", "progress": 0,
            "stage": "queued", "message": "等待处理", "updated_at": timestamp(),
            "archive": str(archive_path(row["archive"])),
            "output": str(output_root / "samples" / sample / "corrected_compartments.h5ad"),
        })


class Status:
    def __init__(self, path: Path, sample: str, archive: Path, output: Path):
        self.path, self.sample, self.archive, self.output = path, sample, archive, output
        self.stage = "queued"
        self.progress = 0
        self.message = "等待处理"
        self.started = timestamp()
        self.current_status = "running"
        self.started_monotonic = time.monotonic()
        self.lock = threading.Lock()
        self.extra: dict = {}

    def update(self, stage: str | None = None, progress: int | None = None,
               message: str | None = None, status: str | None = None, **extra) -> None:
        with self.lock:
            if stage is not None:
                self.stage = stage
            if progress is not None:
                self.progress = progress
            if message is not None:
                self.message = message
            if status is not None:
                self.current_status = status
            self.extra.update(extra)
            atomic_json(self.path, {
                "sample_id": self.sample, "status": self.current_status,
                "progress": self.progress, "stage": self.stage,
                "message": self.message, "updated_at": timestamp(),
                "started_at": self.started,
                "elapsed_seconds": int(time.monotonic() - self.started_monotonic),
                "archive": str(self.archive), "output": str(self.output),
                **self.extra,
            })


class Heartbeat:
    def __init__(self, status: Status, interval: int = 20):
        self.status = status
        self.stop = threading.Event()
        self.interval = interval
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(self.interval):
            self.status.update()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


class TranscriptProgress:
    """Preserve converter logs while exposing its processed-row counter."""

    def __init__(self, stream, log, status: Status):
        self.stream, self.log, self.status = stream, log, status
        self.pending = ""

    def write(self, value: str) -> int:
        self.stream.write(value)
        self.log.write(value)
        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            match = TRANSCRIPT_PATTERN.search(line)
            if match:
                rows, retained = map(int, match.groups())
                self.status.update(message=f"已读取 {rows:,} 条转录本，保留 {retained:,} 条",
                                   processed_rows=rows, retained_transcripts=retained)
        return len(value)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()


def sparse_equal(a, b) -> bool:
    difference = sp.csr_matrix(a) - sp.csr_matrix(b)
    difference.eliminate_zeros()
    return difference.nnz == 0


def archive_members(archive: Path) -> list[dict]:
    relevant = {"cell_feature_matrix.h5", "cells.csv.gz",
                "transcripts.csv.gz", "transcripts.parquet"}
    with zipfile.ZipFile(archive) as handle:
        return [{"member": item.filename, "bytes": item.file_size, "crc32": f"{item.CRC:08x}"}
                for item in handle.infolist() if Path(item.filename).name in relevant]


def combine_compartments(raw_path: Path, corrected_path: Path, final_path: Path,
                         sample: str, archive: Path, manifest_archive: str) -> dict:
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    raw_data = ad.read_h5ad(raw_path)
    corrected_data = ad.read_h5ad(corrected_path)
    if (raw_data.shape != corrected_data.shape
            or not raw_data.obs_names.equals(corrected_data.obs_names)
            or not raw_data.var_names.equals(corrected_data.var_names)):
        raise ValueError("ResolVI output changed cell/gene identity or order")
    raw_total = sp.csr_matrix(raw_data.X, dtype=np.int64)
    raw_nuclear = sp.csr_matrix(raw_data.layers["nuclear_counts"], dtype=np.int64)
    raw_extranuclear = sp.csr_matrix(raw_data.layers["extranuclear_counts"], dtype=np.int64)
    if not sparse_equal(raw_total, raw_nuclear + raw_extranuclear):
        raise ValueError("Raw compartments fail total-count conservation")
    if not sparse_equal(raw_total, corrected_data.layers["raw_total_counts"]):
        raise ValueError("ResolVI raw count layer differs from the archive aggregation")
    if not np.array_equal(raw_data.obsm["spatial"], corrected_data.obsm["spatial"]):
        raise ValueError("ResolVI changed cell spatial coordinates")
    corrected_total = sp.csr_matrix(corrected_data.X, dtype=np.float64)
    corrected_total.eliminate_zeros()
    if not np.isfinite(corrected_total.data).all() or (corrected_total.data < 0).any():
        raise ValueError("Corrected counts contain invalid values")
    raw_nonzero = raw_total.copy().astype(np.float64)
    raw_nonzero.eliminate_zeros()
    support = raw_nonzero.copy()
    support.data.fill(1.0)
    if not sparse_equal(corrected_total, corrected_total.multiply(support)):
        raise ValueError("ResolVI created a nonzero at an originally zero cell-gene position")
    fraction = raw_nuclear.astype(np.float64).multiply(raw_nonzero.power(-1))
    corrected_nuclear = corrected_total.multiply(fraction).tocsr()
    corrected_extranuclear = (corrected_total - corrected_nuclear).tocsr()
    reconstructed = (corrected_nuclear + corrected_extranuclear).tocsr()
    discrepancy = (reconstructed - corrected_total).tocsr()
    discrepancy.eliminate_zeros()
    max_error = float(np.abs(discrepancy.data).max(initial=0.0))
    if max_error > 1e-10 or (corrected_extranuclear.data < 0).any():
        raise ValueError(f"Corrected compartment sum differs from total: {max_error}")
    # X is rebuilt from the two estimated compartments, so this invariant is
    # represented directly, subject only to HDF5 floating-point round-trip.
    final_total = (corrected_nuclear + corrected_extranuclear).tocsr()
    result = ad.AnnData(X=final_total, obs=raw_data.obs.copy(), var=raw_data.var.copy())
    result.obs["resolvi_corrected"] = corrected_data.obs["resolvi_corrected"].to_numpy()
    result.obsm["spatial"] = np.asarray(raw_data.obsm["spatial"]).copy()
    result.layers["nuclear_counts"] = corrected_nuclear
    result.layers["extranuclear_counts"] = corrected_extranuclear
    result.layers["raw_total_counts"] = raw_total
    result.layers["raw_nuclear_counts"] = raw_nuclear
    result.layers["raw_extranuclear_counts"] = raw_extranuclear
    result.uns["compartment_qc"] = raw_data.uns["compartment_qc"]
    result.uns["compartment_source"] = raw_data.uns["compartment_source"]
    result.uns["resolvi_correction"] = corrected_data.uns["resolvi_correction"]
    result.uns["raw_resolvi_pipeline"] = {
        "sample_id": sample, "archive": str(archive.resolve()),
        "manifest_archive": manifest_archive,
        "archive_bytes": archive.stat().st_size,
        "archive_members_json": json.dumps(archive_members(archive), sort_keys=True),
        "raw_compartments": str(raw_path.resolve()),
        "corrected_total": str(corrected_path.resolve()),
        "method": "corrected_nuclear = ResolVI_corrected_total * raw_nuclear / raw_total; corrected_extranuclear = corrected_total - corrected_nuclear",
        "scope": "ResolVI corrects cell-by-gene totals only; compartment values are proportional estimates. No molecule, cell, or nuclear assignment was corrected.",
        "units": "nucleus is 10x overlaps_nucleus; extranuclear is segmentation-based and may include membrane",
        "created_at": timestamp(),
    }
    stage = final_path.with_name(final_path.name + ".partial")
    if stage.exists():
        raise FileExistsError(f"Refusing to overwrite {stage}")
    result.write_h5ad(stage, compression="gzip")
    os.rename(stage, final_path)
    reread = ad.read_h5ad(final_path, backed="r")
    try:
        if (reread.shape != result.shape or not reread.obs_names.equals(result.obs_names)
                or not reread.var_names.equals(result.var_names)
                or not np.array_equal(reread.obsm["spatial"], result.obsm["spatial"])):
            raise ValueError("Final H5AD failed identity/position round-trip")
        if not {"nuclear_counts", "extranuclear_counts", "raw_total_counts",
                "raw_nuclear_counts", "raw_extranuclear_counts"}.issubset(reread.layers):
            raise ValueError("Final H5AD lost one or more count layers")
        for start in range(0, reread.n_obs, 4096):
            stop = min(start + 4096, reread.n_obs)
            if not sparse_equal(reread.X[start:stop],
                                reread.layers["nuclear_counts"][start:stop]
                                + reread.layers["extranuclear_counts"][start:stop]):
                raise ValueError(f"Written corrected layers fail exact sum at rows {start}:{stop}")
            if not sparse_equal(reread.layers["raw_total_counts"][start:stop],
                                reread.layers["raw_nuclear_counts"][start:stop]
                                + reread.layers["raw_extranuclear_counts"][start:stop]):
                raise ValueError(f"Written raw layers fail exact sum at rows {start}:{stop}")
    finally:
        reread.file.close()
    qc = {
        "sample_id": sample, "cells": result.n_obs, "genes": result.n_vars,
        "raw_total": int(raw_total.sum()),
        "raw_nuclear": int(raw_nuclear.sum()),
        "raw_extranuclear": int(raw_extranuclear.sum()),
        "corrected_total": float(final_total.sum()),
        "corrected_nuclear": float(corrected_nuclear.sum()),
        "corrected_extranuclear": float(corrected_extranuclear.sum()),
        "max_reconstruction_error": max_error,
        "corrected_total_difference_from_resolvi": float(abs(final_total - corrected_total).sum()),
        "written_raw_sum_exact": True,
        "written_corrected_sum_exact": True,
        "output": str(final_path), "validated_at": timestamp(),
    }
    atomic_json(final_path.with_suffix(".qc.json"), qc)
    return qc


def run_resolvi(input_path: Path, output_dir: Path, args: argparse.Namespace,
                status: Status, log_path: Path) -> None:
    command = [str(args.python), str(ROOT / "scripts/repair_xenium_total_resolvi.py"),
               "--input", str(input_path), "--output-dir", str(output_dir),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--posterior-chunk", str(args.posterior_chunk),
               "--posterior-samples", str(args.posterior_samples),
               "--min-counts", str(args.min_counts), "--neighbors", str(args.neighbors),
               "--seed", str(args.seed)]
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    status.update("resolvi_training", 35, "正在训练 ResolVI")
    lines: queue.Queue[str | None] = queue.Queue()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None

        def read_lines():
            for line in process.stdout:
                lines.put(line)
            lines.put(None)

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        last_line = "正在训练 ResolVI"
        while True:
            try:
                line = lines.get(timeout=20)
            except queue.Empty:
                status.update(message=last_line)
                continue
            if line is None:
                break
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
            last_line = line.strip()[-240:] or last_line
            match = POSTERIOR_PATTERN.search(line)
            if match:
                done, total = map(int, match.groups())
                status.update("resolvi_posterior", min(85, 70 + int(15 * done / total)),
                              f"后验计算：{done:,}/{total:,} 个细胞",
                              processed_cells=done, eligible_cells=total)
            elif "model trained and saved" in line:
                status.update("resolvi_posterior", 70, "模型已训练，正在计算校正矩阵")
            elif (epoch := EPOCH_PATTERN.search(line)):
                done, total = map(int, epoch.groups())
                status.update("resolvi_training", min(69, 35 + int(34 * done / total)),
                              f"ResolVI epoch {done}/{total}")
        reader.join()
        exit_code = process.wait()
    if exit_code:
        raise RuntimeError(f"ResolVI exited {exit_code}; see {log_path}")
    expected = output_dir / "corrected_total_counts.h5ad"
    if not expected.is_file() or not (output_dir / "qc.json").is_file():
        raise RuntimeError(f"ResolVI exited successfully but output is incomplete: {output_dir}")


def run_sample(row: dict[str, str], output_root: Path, args: argparse.Namespace) -> None:
    sample = row["sample"]
    archive = (args.archive or archive_path(row["archive"])).resolve(strict=True)
    if args.scratch_dir:
        scratch_dir = args.scratch_dir.resolve(strict=True)
        if not scratch_dir.is_dir():
            raise NotADirectoryError(scratch_dir)
        args.scratch_dir = scratch_dir
        # The archive matrix converter uses tempfile's default directory;
        # direct its extracted H5 there along with the transcript Parquet.
        os.environ["TMPDIR"] = str(scratch_dir)
        tempfile.tempdir = str(scratch_dir)
    sample_dir = (output_root / "samples" / sample if args.attempt == "initial"
                  else output_root / "attempts" / sample / args.attempt)
    if sample_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing sample run: {sample_dir}")
    sample_dir.mkdir(parents=True, exist_ok=False)
    final_path = sample_dir / "corrected_compartments.h5ad"
    status = Status(output_root / "state" / f"{sample}.json", sample, archive, final_path)
    try:
        with Heartbeat(status):
            status.update("raw_reference", 5, "从 outs.zip 的原始细胞矩阵生成参考")
            reference_tmp = sample_dir / "raw_reference.partial.h5ad"
            reference = sample_dir / "raw_reference.h5ad"
            convert_xenium_zip_file(archive, reference_tmp, dataset_id=sample,
                                    source="10x_xenium_raw_bundle", sample_id=sample,
                                    batch_id=sample, overwrite=False)
            if not reference_tmp.is_file():
                raise RuntimeError("Raw reference conversion produced no H5AD")
            os.rename(reference_tmp, reference)
            status.update("transcript_aggregation", 18,
                          "逐批读取 outs.zip 的转录本，核对每个细胞×基因的计数")
            compartments = sample_dir / "raw_compartments.h5ad"
            with (sample_dir / "aggregation.log").open("x", encoding="utf-8") as log:
                with redirect_stdout(TranscriptProgress(sys.stdout, log, status)):
                    aggregation_qc = convert_xenium_compartments(
                        archive, reference, compartments, chunk_rows=args.chunk_rows,
                        qv_min=args.qv_min, scratch_dir=args.scratch_dir)
            status.update("raw_counts_validated", 34,
                          "原始核内＋核外计数已与压缩包矩阵逐项核对",
                          rows_seen=aggregation_qc["rows_seen"],
                          retained_transcripts=aggregation_qc["retained"])
            resolvi_dir = sample_dir / "resolvi"
            run_resolvi(compartments, resolvi_dir, args, status,
                        sample_dir / "resolvi.log")
            status.update("final_matrix", 88, "生成同细胞的总计数、核内和核外矩阵")
            qc = combine_compartments(compartments,
                                      resolvi_dir / "corrected_total_counts.h5ad",
                                      final_path, sample, archive, row["archive"])
            status.update("validated", 100, "处理完成并通过计数和位置验证",
                          status="completed", cells=qc["cells"], genes=qc["genes"],
                          corrected_total=qc["corrected_total"])
    except BaseException as exc:
        status.update(message=f"{type(exc).__name__}: {exc}", status="failed")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initialize", action="store_true", help="Create monitoring manifest and pending states")
    group.add_argument("--sample", help="Process exactly one manifest sample")
    parser.add_argument("--archive", type=Path,
                        help="Staged copy of this sample's outs.zip on the compute cluster")
    parser.add_argument("--attempt", default="initial",
                        help="Unique attempt identifier; preserves failed prior outputs")
    parser.add_argument("--finalize-existing", type=Path,
                        help="Finalize validated raw and ResolVI H5ADs from this prior attempt without retraining")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python with scviva-tools==0.1.7 for ResolVI")
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=500_000)
    parser.add_argument("--qv-min", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--posterior-chunk", type=int, default=1024)
    parser.add_argument("--posterior-samples", type=int, default=30)
    parser.add_argument("--min-counts", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    source_manifest = args.manifest.resolve(strict=True)
    rows = load_manifest(source_manifest)
    output_root = args.output_root.resolve()
    initialize(rows, output_root, source_manifest)
    if args.initialize:
        print(f"[initialized] {len(rows)} samples at {output_root}")
        return
    if args.sample not in {row["sample"] for row in rows}:
        parser.error(f"Sample is absent from manifest: {args.sample}")
    if args.chunk_rows <= 0 or args.qv_min < 0:
        parser.error("chunk-rows must be positive and qv-min nonnegative")
    if not SAMPLE_PATTERN.fullmatch(args.attempt):
        parser.error("attempt must be a safe identifier")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Processing requires a verified Slurm compute-node allocation")
    row = next(row for row in rows if row["sample"] == args.sample)
    if args.finalize_existing:
        prior = args.finalize_existing.resolve(strict=True)
        if not prior.is_dir() or not prior.is_relative_to(output_root):
            parser.error("finalize-existing must be a prior attempt directory under output-root")
        raw_path = prior / "raw_compartments.h5ad"
        corrected_path = prior / "resolvi/corrected_total_counts.h5ad"
        if not raw_path.is_file() or not corrected_path.is_file():
            parser.error("prior attempt lacks validated raw and corrected-total H5ADs")
        final_dir = output_root / "finalized" / args.sample / args.attempt
        if final_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {final_dir}")
        final_dir.mkdir(parents=True, exist_ok=False)
        final_path = final_dir / "corrected_compartments.h5ad"
        archive = (args.archive or archive_path(row["archive"])).resolve(strict=True)
        status = Status(output_root / "state" / f"{args.sample}.json", args.sample,
                        archive, final_path)
        try:
            status.update("final_matrix", 88, "复用已完成的 ResolVI 输出，验证并写入核／质矩阵")
            qc = combine_compartments(raw_path, corrected_path, final_path,
                                      args.sample, archive, row["archive"])
            status.update("validated", 100, "处理完成并通过计数和位置验证",
                          status="completed", cells=qc["cells"], genes=qc["genes"],
                          corrected_total=qc["corrected_total"])
        except BaseException as exc:
            status.update(message=f"{type(exc).__name__}: {exc}", status="failed")
            raise
        return
    run_sample(row, output_root, args)


if __name__ == "__main__":
    main()
