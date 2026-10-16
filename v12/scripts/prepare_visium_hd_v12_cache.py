#!/usr/bin/env python3
"""Build immutable sparse raw-count HD caches for the unchanged V12 encoder.

Input is the validated output of prepare_visium_hd_compartments.py. This does
not infer nuclear localization, invent cytoplasmic expression, or use labels.
Execute on a Slurm compute node; no historical source/cache is overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import h5py
import numpy as np
from scipy import sparse
import yaml

from prepare_subcellular_v2 import prepare_patches, spatial_split
from prepare_unified_training_cache import build_geometry
from spatial_pretrain.data.subcellular_v2 import stable_seed, vocabulary_path
from spatial_pretrain.data.visium_hd_mix import (
    VISIUM_CACHE_SCHEMA, VisiumHDV12Dataset, assert_not_crc_source, file_sha256,
    physical_sample_key,
)


def require_compute_node():
    host = socket.gethostname().split(".")[0]
    node = os.getenv("SLURMD_NODENAME", "").split(".")[0]
    if not os.getenv("SLURM_JOB_ID") or not node or host != node:
        raise RuntimeError("Sparse HD cache preparation requires an actual Slurm compute-node step")


def _text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def read_strings(element):
    if isinstance(element, h5py.Dataset):
        values = element[:]
    else:
        codes = np.asarray(element["codes"][:], dtype=np.int64)
        if (codes < 0).any():
            raise ValueError("Missing source gene/sample/species identifier")
        values = element["categories"][:][codes]
    return np.asarray([_text(value) for value in values], dtype=str)


def read_inputs_manifest(path):
    path = Path(path).resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"sample_id", "physical_sample_id", "compartments_h5ad"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"Inputs TSV needs columns {sorted(required)}")
        rows = []
        samples, physical_samples, paths = set(), set(), set()
        for line, record in enumerate(reader, 2):
            record = {key: (value or "").strip() for key, value in record.items() if key is not None}
            if any(not record.get(field) for field in required):
                raise ValueError(f"Inputs TSV line {line}: missing required identifier/path")
            assert_not_crc_source(record)
            physical_key = physical_sample_key(record["physical_sample_id"])
            source = Path(record["compartments_h5ad"]).expanduser()
            source = (source if source.is_absolute() else path.parent / source).resolve()
            if (record["sample_id"] in samples or physical_key in physical_samples
                    or source in paths):
                raise ValueError("Duplicate HD sample ID, physical sample ID or source path")
            if not source.is_file():
                raise FileNotFoundError(source)
            samples.add(record["sample_id"])
            physical_samples.add(physical_key)
            paths.add(source)
            record["compartments_h5ad"] = str(source)
            rows.append(record)
    if not rows:
        raise ValueError("Inputs manifest has no samples")
    return rows


def mapped_measured_panel(gene_symbols, vocab):
    """Exact symbols only; duplicate measured features sum as raw counts."""
    names = np.asarray(gene_symbols, dtype=str)
    columns = np.flatnonzero([name in vocab and not name.startswith("<") for name in names])
    if not len(columns):
        raise ValueError("No measured genes match the ordinary scGPT vocabulary")
    feature_ids = np.asarray([vocab[names[column]] for column in columns], dtype=np.int64)
    genes, inverse = np.unique(feature_ids, return_inverse=True)
    projector = sparse.csr_matrix((np.ones(len(columns), dtype=np.int64), (columns, inverse)),
                                 shape=(len(names), len(genes)))
    symbols = []
    for gene in genes:
        aliases = np.unique(names[columns[feature_ids == gene]])
        if len(aliases) != 1:
            raise ValueError("Different gene symbols alias one vocabulary ID")
        symbols.append(str(aliases[0]))
    audit = dict(input_measured_features=len(names), mapped_measured_features=len(columns),
                 mapped_unique_genes=len(genes), duplicate_mapped_features_summed=len(columns) - len(genes),
                 unmatched_features_excluded=int(len(names) - len(columns)),
                 observed_zero_genes_retained=True, full_vocab_zero_filling=False,
                 mapping="exact gene_name to vocabulary; duplicate symbols sum raw N and C separately")
    return genes, symbols, projector, columns, audit


def _integer_counts(values, label):
    values = np.asarray(values)
    if (values.dtype.kind not in "iuf" or not np.isfinite(values).all() or (values < 0).any()
            or (values >= 2**53).any()
            or (values.dtype.kind == "f" and not np.array_equal(values, np.rint(values)))):
        raise ValueError(f"{label}: raw nonnegative exact integer counts below 2**53 are required")
    return values.astype(np.int64, copy=False)


def read_raw_rows(element, start, stop, genes):
    if isinstance(element, h5py.Dataset):
        matrix = sparse.csr_matrix(_integer_counts(element[start:stop], element.name))
    else:
        if _text(element.attrs.get("encoding-type", "")) != "csr_matrix":
            raise ValueError(f"{element.name}: lossless CSR or dense source required")
        shape = tuple(int(value) for value in element.attrs["shape"])
        if shape[1] != genes or stop > shape[0]:
            raise ValueError("Source count matrix shape differs from observation/feature axes")
        pointer = np.asarray(element["indptr"][start:stop + 1], dtype=np.int64)
        first, last = int(pointer[0]), int(pointer[-1])
        values = _integer_counts(element["data"][first:last], element.name)
        indices = np.asarray(element["indices"][first:last], dtype=np.int64)
        if ((indices < 0) | (indices >= genes)).any():
            raise ValueError("Source sparse gene index outside measured panel")
        matrix = sparse.csr_matrix((values, indices, pointer - first), shape=(stop - start, genes))
    if matrix.shape != (stop - start, genes):
        raise ValueError("Source matrix rows/features differ from source coordinate/panel axes")
    matrix.check_format(full_check=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    _integer_counts(matrix.data, element.name)
    return matrix


def exact_row_sums(matrix):
    maximum = int(matrix.data.max(initial=0))
    if maximum * max(matrix.shape[1], matrix.nnz) >= 2**63:
        raise ValueError("Count sums could overflow int64; explicit wider handling is required")
    return np.asarray(matrix.sum(axis=1)).ravel().astype(np.int64)


class SparseCountWriter:
    """Append canonical CSR chunks; every physical row gets a row pointer."""

    def __init__(self, folder, prefix, shape):
        self.folder, self.prefix, self.shape = Path(folder), prefix, tuple(shape)
        self.pointer = np.lib.format.open_memmap(self.folder / f"{prefix}.indptr.npy",
            mode="w+", dtype=np.int64, shape=(self.shape[0] + 1,))
        self.pointer[0] = 0
        self.data = (self.folder / f"{prefix}.data.bin").open("xb")
        self.indices = (self.folder / f"{prefix}.indices.bin").open("xb")
        self.rows, self.nnz = 0, 0

    def append(self, matrix):
        matrix = sparse.csr_matrix(matrix, dtype=np.int64)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
        if matrix.shape[1] != self.shape[1] or self.rows + matrix.shape[0] > self.shape[0]:
            raise ValueError("Sparse cache append shape mismatch")
        _integer_counts(matrix.data, "mapped cache counts")
        if self.shape[1] > np.iinfo(np.int32).max:
            raise ValueError("Measured panel exceeds int32 sparse column capacity")
        self.data.write(np.asarray(matrix.data, dtype="<i8").tobytes())
        self.indices.write(np.asarray(matrix.indices, dtype="<i4").tobytes())
        self.pointer[self.rows + 1:self.rows + matrix.shape[0] + 1] = matrix.indptr[1:] + self.nnz
        self.rows += matrix.shape[0]
        self.nnz += matrix.nnz

    def close(self, *, complete=True):
        self.data.close()
        self.indices.close()
        self.pointer.flush()
        if complete and self.rows != self.shape[0]:
            raise ValueError("Sparse writer ended before every source row was appended")


def provenance_strings(group):
    """Only small provenance datasets; never walk source count arrays."""
    values = []
    for name, element in group.items():
        if isinstance(element, h5py.Group):
            values.extend(provenance_strings(element))
        elif element.dtype.kind in "OSU" and element.size <= 1000:
            values.extend(_text(value) for value in np.asarray(element[()]).ravel())
    return values


def prepare_sample(record, folder, config, vocab):
    source, folder = Path(record["compartments_h5ad"]), Path(folder)
    audit_path = source.parent / "audit.json"
    source_audit = json.loads(audit_path.read_text())
    checks = source_audit.get("checks", {})
    if source_audit.get("status") != "complete" or checks.get("X_equals_nuclear_plus_cytoplasmic") is not True:
        raise ValueError("HD source lacks a completed compartment conservation audit")
    if Path(source_audit["output_h5ad"]).name != source.name:
        raise ValueError("Compartment audit names a different H5AD")
    assert_not_crc_source(record, source_audit)
    folder.mkdir(parents=True, exist_ok=False)
    writers = []
    try:
        with h5py.File(source, "r") as handle:
            schema = _text(handle["uns/input_schema"][()])
            if schema != "visium_hd_segmented_compartments_v1":
                raise ValueError("Expected actual Space Ranger image/bin-derived HD compartment schema")
            provenance = provenance_strings(handle["uns/compartment_provenance"])
            assert_not_crc_source(provenance)
            species = record.get("species", "")
            if not species and "species" in handle["obs"]:
                observed_species = np.unique(read_strings(handle["obs/species"]))
                if len(observed_species) == 1:
                    species = str(observed_species[0])
            if species != "Homo sapiens":
                raise ValueError("Explicit species='Homo sapiens' is required for human scGPT ingestion")
            if "sample_id" in handle["obs"]:
                source_samples = np.unique(read_strings(handle["obs/sample_id"]))
                if len(source_samples) != 1:
                    raise ValueError("One physical HD sample per source H5AD is required")
                assert_not_crc_source(source_samples.tolist())
            names = read_strings(handle["var/gene_name"])
            if "measured" not in handle["var"] or not np.asarray(handle["var/measured"][:], dtype=bool).all():
                raise ValueError("Every source column must be declared truly measured")
            genes, gene_names, projector, matched_columns, gene_audit = mapped_measured_panel(names, vocab)
            coords = np.asarray(handle["obsm/spatial_um"], dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all() or len(coords) < 2:
                raise ValueError("Finite physical micrometer coordinates and at least two cells are required")
            obs_index = _text(handle["obs"].attrs.get("_index", "_index"))
            cell_ids = read_strings(handle["obs"][obs_index])
            if len(cell_ids) != len(coords) or len(np.unique(cell_ids)) != len(cell_ids):
                raise ValueError("Source physical cell IDs must be unique and align to coordinates")
            if "coordinate_units" not in handle["uns/compartment_provenance"] or _text(
                    handle["uns/compartment_provenance/coordinate_units"][()]) != "micrometres":
                raise ValueError("Source coordinate units are not validated micrometers")
            counts_shape = (len(coords), len(genes))
            writers = [SparseCountWriter(folder, prefix, counts_shape) for prefix in ("nuclear", "cytoplasmic")]
            total_per_cell = np.zeros(len(coords), dtype=np.int64)
            mapped_umi = nuclear_umi = cyto_umi = source_umi = 0
            for start in range(0, len(coords), 512):
                stop = min(start + 512, len(coords))
                nuclear = read_raw_rows(handle["layers/nuclear_counts"], start, stop, len(names))
                cyto = read_raw_rows(handle["layers/cytoplasmic_counts"], start, stop, len(names))
                total = read_raw_rows(handle["X"], start, stop, len(names))
                if (nuclear + cyto != total).nnz:
                    raise ValueError("Source N + C differs from X before any gene mapping")
                mapped_n = (nuclear @ projector).tocsr()
                mapped_c = (cyto @ projector).tocsr()
                expected = exact_row_sums(total[:, matched_columns])
                n_sum, c_sum = exact_row_sums(mapped_n), exact_row_sums(mapped_c)
                if not np.array_equal(n_sum + c_sum, expected):
                    raise ValueError("Duplicate-symbol/raw-count projection changed measured counts")
                total_per_cell[start:stop] = expected
                mapped_umi += int(expected.sum())
                nuclear_umi += int(n_sum.sum())
                cyto_umi += int(c_sum.sum())
                source_umi += int(exact_row_sums(total).sum())
                writers[0].append(mapped_n)
                writers[1].append(mapped_c)
                if start == 0 or stop == len(coords) or start % (512 * 200) == 0:
                    print(f"[sparse-HD] {record['sample_id']} rows={stop}/{len(coords)} genes={len(genes)}/{len(names)}", flush=True)
            for writer in writers:
                writer.close()
            np.save(folder / "total_counts_per_cell.npy", total_per_cell)
            np.save(folder / "gene_ids.npy", genes)
            (folder / "gene_names.txt").write_text("\n".join(gene_names) + "\n")
            (folder / "cell_ids.txt").write_text("\n".join(cell_ids) + "\n")
            np.save(folder / "coords.npy", coords)
        # The physical graph includes all source cells before center selection.
        groups = np.full(len(coords), record["physical_sample_id"], dtype=object)
        knn = build_geometry(coords, groups, k=255)
        np.save(folder / "knn_indices.npy", knn)
        data = dict(config["data"])
        data.pop("whole_cell_max_tokens", None)
        train = spatial_split(coords, data)
        np.save(folder / "train_cell_mask.npy", train)
        metadata = dict(sample_id=record["sample_id"], physical_sample_id=record["physical_sample_id"],
            species=species, source_h5ad=str(source.resolve()), source_h5ad_sha256=file_sha256(source),
            source_audit_sha256=file_sha256(audit_path), cache_dir=str(folder.resolve()),
            num_cells=len(coords), num_genes=len(genes), nuclear_nnz=writers[0].nnz,
            cytoplasmic_nnz=writers[1].nnz, measured_panel=gene_audit,
            source_measured_umi=source_umi, mapped_measured_umi=mapped_umi,
            nuclear_umi=nuclear_umi, cytoplasmic_umi=cyto_umi,
            coordinate_unit="micrometer", graph="full_source_cells_KNN255_then_split_isolated_fixed_64cell_patches",
            compartment_limitation="2um capture-bin/image-derived compartments; not molecule-resolved localization",
            mapped_empty_cells=int((total_per_cell == 0).sum()),
            checks=dict(full_source_N_plus_C_equals_X=True, duplicate_symbol_mapping_conservation=True,
                        no_full_vocab_zero_filling=True, CRC_benchmark_excluded=True))
        for split, members in (("train", train), ("val", ~train)):
            centers, nodes, stats = prepare_patches(knn, members, int(data["context_cells"]),
                int(data["center_stride"]), stable_seed(f"{data['seed']}:{record['sample_id']}:{split}"))
            keep = total_per_cell[centers] > 0
            center_filter = dict(original_centers=len(centers), retained_centers=int(keep.sum()),
                                 excluded_centers=int((~keep).sum()), excluded_cell_ids=centers[~keep].tolist())
            np.save(folder / f"{split}_centers.npy", centers[keep])
            np.save(folder / f"{split}_nodes.npy", nodes[keep])
            metadata[split] = dict(stats, center_filter=center_filter,
                                  eligible_nonempty_centers=int(keep.sum()))
        metadata["files_sha256"] = {path.name: file_sha256(path) for path in sorted(folder.iterdir()) if path.is_file()}
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata
    except BaseException:
        for writer in writers:
            if not writer.data.closed:
                writer.close(complete=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require_compute_node()
    config = yaml.safe_load(args.config.read_text())
    if config.get("binning") != {"num_bins": 51, "tie_policy": "left"}:
        raise ValueError("The source V12 deterministic-left 51-bin config is required")
    if config["model"].get("backbone_kind") != "scgpt":
        raise ValueError("Only ordinary human scGPT ingestion is supported")
    rows = read_inputs_manifest(args.inputs_manifest)
    vocab_path = vocabulary_path(config["assets_dir"])
    vocab = json.loads(vocab_path.read_text())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        slices, digests = [], set()
        for index, record in enumerate(rows):
            digest = file_sha256(record["compartments_h5ad"])
            if digest in digests:
                raise ValueError("The same physical count file occurs twice under different names")
            digests.add(digest)
            slices.append(prepare_sample(record, output / f"sample_{index:03d}", config, vocab))
        data = dict(config["data"])
        data.pop("whole_cell_max_tokens", None)
        manifest = dict(schema=VISIUM_CACHE_SCHEMA, status="validated", data_config=data,
            vocab_sha256=file_sha256(vocab_path), inputs_manifest_sha256=file_sha256(args.inputs_manifest),
            slice_count=len(slices), slices=slices, total_cells=sum(item["num_cells"] for item in slices),
            sparse_count_bytes=sum((Path(item["cache_dir"]) / filename).stat().st_size
                for item in slices for filename in item["files_sha256"] if filename.endswith(".bin")),
            dense_float32_compartment_bytes_avoided=sum(item["num_cells"] * item["num_genes"] * 8 for item in slices),
            CRC_evaluation_sample_excluded=True, source_hashes_verified=True,
            representation="per_sample_full_measured_panel_CSR_int64_counts",
            training_started=False)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        runtime = dict(config, visium_cache_root=str(output))
        loader_checks = {}
        for split in ("train", "val"):
            dataset = VisiumHDV12Dataset(runtime, split=split)
            selected = sorted(set([0, len(dataset) // 2, len(dataset) - 1]))
            for index in selected:
                view = dataset[index]
                valid = view["query_valid_mask"]
                if not np.array_equal(view["query_nucleus_counts"] + view["query_cytoplasm_counts"],
                                      view["query_total_counts"]):
                    raise ValueError("Sparse cache real-loader paired targets fail conservation")
                if not valid.any() or int(view["domain_id"]) != 1:
                    raise ValueError("Sparse cache real-loader produced no valid measured queries")
            loader_checks[split] = dict(centers=len(dataset), checked_indices=selected,
                                        samples=list(dataset.indices_by_sample))
        (output / "LOADER_VALIDATED.json").write_text(json.dumps(loader_checks, indent=2) + "\n")
        print(json.dumps(dict(status="validated_sparse_HD_cache", cache=str(output),
                              total_cells=manifest["total_cells"], loader_checks=loader_checks)), flush=True)
    except BaseException as error:
        if (output / "manifest.json").exists():
            (output / "manifest.json").rename(output / "manifest.failed_validation.json")
        (output / "FAILED.json").write_text(json.dumps(dict(status="failed", error_type=type(error).__name__,
                                                           error=str(error)), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
