#!/usr/bin/env python3
"""Read-only, streaming H5AD compartment audit. Run on a compute node.

The output is an ingestion catalog, not a training split. No source H5AD,
sidecar, graph, or training configuration is rewritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from scipy import sparse


SCHEMA = "existing_compartment_corpus_audit_v1"
MAX_EXACT_COUNT = 2**53 - 1
INT64_MAX = np.iinfo(np.int64).max


def text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def integer_counts(values, label):
    values = np.asarray(values)
    if values.dtype.kind not in "biuf":
        raise ValueError(f"{label}: counts must be numeric")
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values > MAX_EXACT_COUNT).any()
            or (values != np.floor(values)).any()):
        raise ValueError(f"{label}: expected finite nonnegative integer counts <= 2**53-1")
    return values.astype(np.int64, copy=False)


def exact_sum(values):
    """Fast normal path, Python-integer fallback for theoretical int64 overflow."""
    if not values.size:
        return 0
    if int(values.max()) * values.size <= INT64_MAX:
        return int(values.sum(dtype=np.int64))
    return sum(int(value) for value in values.flat)


class MatrixRows:
    def __init__(self, element, shape, label):
        self.element, self.shape, self.label = element, tuple(shape), label
        if isinstance(element, h5py.Dataset):
            if element.shape != self.shape:
                raise ValueError(f"{label}: matrix shape {element.shape} != {self.shape}")
            self.csr = False
            return
        encoding = text(element.attrs.get("encoding-type", ""))
        if encoding != "csr_matrix":
            raise ValueError(f"{label}: expected dense/CSR, got {encoding!r}")
        if tuple(element.attrs.get("shape", ())) != self.shape:
            raise ValueError(f"{label}: CSR shape does not match axes")
        for name in ("data", "indices", "indptr"):
            if name not in element or element[name].ndim != 1:
                raise ValueError(f"{label}: missing or invalid CSR {name}")
        if (element["indices"].dtype.kind not in "iu"
                or element["indptr"].dtype.kind not in "iu"):
            raise ValueError(f"{label}: CSR indices/pointers must be integers")
        nnz = len(element["data"])
        if (len(element["indices"]) != nnz
                or len(element["indptr"]) != self.shape[0] + 1
                or int(element["indptr"][0]) != 0
                or int(element["indptr"][-1]) != nnz):
            raise ValueError(f"{label}: invalid CSR lengths/pointer endpoints")
        self.csr = True

    def read(self, start, stop):
        if not self.csr:
            return integer_counts(self.element[start:stop], self.label)
        group = self.element
        ptr = np.asarray(group["indptr"][start:stop + 1], dtype=np.int64)
        if ((ptr < 0).any() or (np.diff(ptr) < 0).any()
                or (ptr > len(group["data"])).any()):
            raise ValueError(f"{self.label}: invalid CSR row pointers")
        first, last = int(ptr[0]), int(ptr[-1])
        indices = np.asarray(group["indices"][first:last], dtype=np.int64)
        if (indices < 0).any() or (indices >= self.shape[1]).any():
            raise ValueError(f"{self.label}: CSR column out of bounds")
        data = integer_counts(group["data"][first:last], self.label)
        # Prevent duplicate CSR entries from overflowing during densification.
        if data.size and int(data.max()) * data.size > INT64_MAX:
            raise ValueError(f"{self.label}: possible CSR duplicate count overflow")
        matrix = sparse.csr_matrix((data, indices, ptr - first),
                                   shape=(stop - start, self.shape[1]))
        return integer_counts(matrix.toarray(), self.label)


def read_axis(group, label):
    index_name = text(group.attrs.get("_index", "_index"))
    if index_name not in group or not isinstance(group[index_name], h5py.Dataset):
        raise ValueError(f"{label}: missing plain axis index")
    raw = group[index_name]
    if raw.ndim != 1:
        raise ValueError(f"{label}: axis index must be one-dimensional")
    names = [text(value) for value in raw[:]]
    if not names or any(not name for name in names):
        raise ValueError(f"{label}: empty axis or identifier")
    if len(set(names)) != len(names):
        raise ValueError(f"{label}: identifiers must be unique")
    return names


def vocab_names(path):
    if path is None:
        return None, None
    payload = Path(path).read_bytes()
    vocab = json.loads(payload)
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Vocabulary must be a nonempty gene-to-token dictionary")
    return set(vocab) - {"<pad>", "<cls>", "<eoc>"}, hashlib.sha256(payload).hexdigest()


def cache_mapping(path):
    if path is None:
        return {}
    import yaml
    config = yaml.safe_load(Path(path).read_text())
    data = config["data"]
    sources, caches = data["h5ad_paths"], data["cache_dirs"]
    if len(sources) != len(caches) or len(set(sources)) != len(sources):
        raise ValueError("Base config needs unique one-to-one source/cache lists")
    return {str(Path(source).resolve()): str(cache)
            for source, cache in zip(sources, caches)}


def mapped_cache(source, mapping):
    exact = mapping.get(str(source.resolve()))
    if exact is not None:
        return exact, "resolved_path"
    matches = [cache for path, cache in mapping.items() if Path(path).name == source.name]
    return (matches[0], "basename_only_unverified") if len(matches) == 1 else (None, "unmapped")


def audit_file(source, *, chunk_rows=512, species=None, platform=None,
               vocab=None, vocab_sha256=None, mapping=None):
    if not 1 <= chunk_rows <= 1024:
        raise ValueError("chunk_rows must lie in [1, 1024]")
    source = Path(source)
    started = time.monotonic()
    stat = source.stat()
    qc_path = source.with_suffix(".qc.json")
    qc_payload = qc_path.read_bytes() if qc_path.exists() else None
    qc = json.loads(qc_payload) if qc_payload is not None else None
    if qc is not None and not isinstance(qc, dict):
        raise ValueError("QC sidecar must contain a JSON object")
    record = {
        "schema": SCHEMA, "dataset_id": source.stem,
        "path": str(source.resolve()), "file_size": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns, "species": species or "unknown",
        "platform": platform or "unknown",
        "species_evidence": "explicit_collection_label" if species else "unverified",
        "platform_evidence": "explicit_collection_label" if platform else "unverified",
        "schema_verified": False, "training_split_ready": False,
        "donor_audited": False, "benchmark_overlap_audited": False,
        "source_qc_path": str(qc_path.resolve()) if qc is not None else None,
        "source_qc_sha256": hashlib.sha256(qc_payload).hexdigest() if qc_payload is not None else None,
        "raw_molecule_integrity_rechecked": False,
        "coordinates_unit": "unverified_source_metadata",
        "geometry_cache_verified": False,
    }
    record["geometry_cache"], record["geometry_cache_mapping"] = mapped_cache(source, mapping or {})
    with h5py.File(source, "r") as handle:
        cells = read_axis(handle["obs"], "obs")
        genes = read_axis(handle["var"], "var")
        shape = (len(cells), len(genes))
        record.update(cells=shape[0], genes=shape[1],
                      panel_sha256=hashlib.sha256(json.dumps(genes, ensure_ascii=False,
                          separators=(",", ":")).encode()).hexdigest())
        if vocab is not None:
            matched = sum(gene in vocab for gene in genes)
            record.update(vocab_sha256=vocab_sha256, vocab_matched_genes=matched,
                          vocab_coverage=matched / len(genes),
                          vocab_unmatched_genes=[gene for gene in genes if gene not in vocab])
        coords = handle["obsm"]["spatial"]
        if not isinstance(coords, h5py.Dataset) or coords.shape != (shape[0], 2):
            raise ValueError("spatial coordinates must match cell rows and have exactly two columns")
        nuclear = MatrixRows(handle["layers"]["nuclear_counts"], shape, "nuclear_counts")
        cyto = MatrixRows(handle["layers"]["extranuclear_counts"], shape, "extranuclear_counts")
        total = MatrixRows(handle["X"], shape, "X")
        nuclear_sum = cyto_sum = zero_cells = 0
        for start in range(0, shape[0], chunk_rows):
            stop = min(start + chunk_rows, shape[0])
            xy = np.asarray(coords[start:stop])
            if xy.dtype.kind not in "biuf" or not np.isfinite(xy).all():
                raise ValueError(f"spatial coordinates must be finite numeric values; rows {start}:{stop}")
            n, c, x = nuclear.read(start, stop), cyto.read(start, stop), total.read(start, stop)
            if (n > INT64_MAX - c).any() or not np.array_equal(n + c, x):
                raise ValueError(f"nuclear_counts + extranuclear_counts != X at rows {start}:{stop}")
            nuclear_sum += exact_sum(n)
            cyto_sum += exact_sum(c)
            zero_cells += int((~(x > 0).any(axis=1)).sum())
            if start == 0 or stop == shape[0] or (start // chunk_rows) % 256 == 0:
                print(f"[audit] {source.name} rows={stop}/{shape[0]} elapsed={time.monotonic()-started:.1f}s", flush=True)
        record.update(nuclear_counts=nuclear_sum, cytoplasmic_counts=cyto_sum,
                      retained_counts=nuclear_sum + cyto_sum, zero_count_cells=zero_cells)
        if nuclear_sum + cyto_sum <= 0:
            raise ValueError("No retained molecules in source")
    checked_qc = []
    for key, actual in (("nuclear", nuclear_sum), ("cytoplasmic", cyto_sum),
                        ("retained", nuclear_sum + cyto_sum)):
        if qc is not None and key in qc:
            expected = qc[key]
            if isinstance(expected, bool) or not isinstance(expected, int) or expected != actual:
                raise ValueError(f"QC sidecar {key}={expected!r} disagrees with matrix count {actual}")
            checked_qc.append(key)
    after = source.stat()
    if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
        raise ValueError("Source file changed during audit")
    if qc_payload is not None and qc_path.read_bytes() != qc_payload:
        raise ValueError("QC sidecar changed during audit")
    record.update(schema_verified=True, qc_count_fields_verified=checked_qc,
                  elapsed_seconds=round(time.monotonic() - started, 3),
                  status="schema_verified_pending_split_and_provenance")
    return record


TSV_FIELDS = ["dataset_id", "path", "file_size", "file_mtime_ns", "species", "platform",
              "cells", "genes", "panel_sha256", "nuclear_counts", "cytoplasmic_counts",
              "retained_counts", "zero_count_cells", "schema_verified", "training_split_ready",
              "source_qc_sha256", "vocab_coverage", "geometry_cache", "geometry_cache_mapping",
              "status", "elapsed_seconds", "error"]


def write_catalog(output, records, summary):
    atomic_json(output / "catalog.json", {"schema": SCHEMA, "summary": summary, "datasets": records})
    temporary = output / "catalog.tsv.partial"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=TSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(output / "catalog.tsv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output refused")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--species", help="Explicit collection label, never inferred from file names")
    parser.add_argument("--platform", help="Explicit collection platform label")
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--vocab", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.chunk_rows <= 1024:
        parser.error("--chunk-rows must lie in [1, 1024]")
    files = sorted(args.source_dir.glob("*.h5ad"))
    if not files:
        parser.error("source directory has no H5AD files")
    vocabulary, vocab_sha = vocab_names(args.vocab)
    mapping = cache_mapping(args.base_config)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "files").mkdir()
    started = time.monotonic()
    summary = {"started_at": datetime.now(timezone.utc).isoformat(),
               "source_dir": str(args.source_dir.resolve()), "total_files": len(files),
               "status": "running", "verified_files": 0, "failed_files": 0,
               "training_split_ready": False, "chunk_rows": args.chunk_rows}
    records = []
    write_catalog(args.output, records, summary)
    for index, source in enumerate(files):
        per_started = time.monotonic()
        try:
            record = audit_file(source, chunk_rows=args.chunk_rows, species=args.species,
                                platform=args.platform, vocab=vocabulary, vocab_sha256=vocab_sha,
                                mapping=mapping)
            summary["verified_files"] += 1
        except Exception as error:
            stat = source.stat()
            record = {"dataset_id": source.stem, "path": str(source.resolve()),
                      "file_size": stat.st_size, "file_mtime_ns": stat.st_mtime_ns,
                      "status": "failed", "schema_verified": False,
                      "training_split_ready": False, "error": f"{type(error).__name__}: {error}",
                      "elapsed_seconds": round(time.monotonic() - per_started, 3)}
            summary["failed_files"] += 1
        records.append(record)
        atomic_json(args.output / "files" / f"{index:04d}_{source.stem}.json", record)
        summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
        write_catalog(args.output, records, summary)
        print(f"[audit-file] {index+1}/{len(files)} {source.name} status={record['status']} "
              f"seconds={record['elapsed_seconds']}", flush=True)
    verified = [record for record in records if record["schema_verified"]]
    summary.update(status="completed" if not summary["failed_files"] else "completed_with_failures",
                   completed_at=datetime.now(timezone.utc).isoformat(),
                   cells=sum(record["cells"] for record in verified),
                   retained_counts=sum(record["retained_counts"] for record in verified),
                   nuclear_counts=sum(record["nuclear_counts"] for record in verified),
                   cytoplasmic_counts=sum(record["cytoplasmic_counts"] for record in verified),
                   elapsed_seconds=round(time.monotonic() - started, 3))
    write_catalog(args.output, records, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not summary["failed_files"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
