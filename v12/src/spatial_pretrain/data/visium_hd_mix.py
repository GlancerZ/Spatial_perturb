"""Sparse raw-count Visium HD views for the unchanged V12 architecture.

The measured panel belongs to each sample. A sparse zero is an observed zero
only within that panel; missing genes are never expanded to the scGPT universe.
Caches store counts rather than dense log-count arrays. Only the small physical
patch used by an example is materialized before the existing V12 encoder runs.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .subcellular_v2 import stable_seed, vocabulary_path
from .subcellular_v6 import load_lr_prior
from .subcellular_v10 import _integer, _neighbor_rows
from .subcellular_v12 import V12Dataset, V12_TARGET_METADATA, encode_v12_views


VISIUM_CACHE_SCHEMA = "visium_hd_v12_sparse_raw_counts_v1"
MIXED_DATASET_SCHEMA = "xenium48_visium_hd_v12_views_v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_sample_key(value):
    key = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    if not key:
        raise ValueError("A nonempty physical sample ID is required")
    return key


def assert_not_crc_source(*values):
    """Reject the CRC benchmark and historical names even after file renaming.

This intentionally fails closed on the Colon_Cancer/Colorectal_Cancer aliases.
The slide/well spelling covers metadata that lacks the public sample name.
Callers supply source provenance as well as caller-provided manifest labels.
"""
    strings = []

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif value is not None:
            strings.append(str(value))

    for value in values:
        visit(value)
    for text in strings:
        normalized = re.sub(r"[^a-z0-9]", "", text.casefold())
        if ("coloncancer" in normalized or "colorectalcancer" in normalized
                or "h1vm2jxxka1" in normalized):
            raise ValueError(f"Excluded CRC evaluation sample or alias: {text!r}")
    joined = " ".join(strings).casefold()
    if re.search(r"h1[-_ ]?vm2jxxk", joined) and re.search(r"\ba1\b", joined):
        raise ValueError("Excluded CRC evaluation slide H1-VM2JXXK, well A1")


class SparseRawCountRows:
    """Read canonical CSR rows without materializing the complete count matrix."""

    def __init__(self, folder, prefix, shape, nnz):
        self.folder, self.prefix = Path(folder), prefix
        self.shape = tuple(int(value) for value in shape)
        if len(self.shape) != 2 or min(self.shape) < 1 or isinstance(nnz, bool) or int(nnz) < 0:
            raise ValueError("Invalid sparse count shape or nnz")
        self.nnz = int(nnz)
        self.indptr = np.load(self.folder / f"{prefix}.indptr.npy", mmap_mode="r")
        if (self.indptr.dtype != np.int64 or self.indptr.shape != (self.shape[0] + 1,)
                or self.indptr[0] != 0 or self.indptr[-1] != self.nnz
                or (np.diff(self.indptr) < 0).any()):
            raise ValueError("Invalid sparse count row pointer")
        data_path, index_path = [self.folder / f"{prefix}.{suffix}.bin" for suffix in ("data", "indices")]
        if data_path.stat().st_size != self.nnz * 8 or index_path.stat().st_size != self.nnz * 4:
            raise ValueError("Sparse count file length disagrees with nnz")
        self.data = (np.memmap(data_path, mode="r", dtype="<i8", shape=(self.nnz,))
                     if self.nnz else np.empty(0, dtype=np.int64))
        self.indices = (np.memmap(index_path, mode="r", dtype="<i4", shape=(self.nnz,))
                        if self.nnz else np.empty(0, dtype=np.int32))

    def dense_rows(self, rows):
        rows = np.asarray(rows)
        if (rows.ndim != 1 or rows.dtype.kind not in "iu"
                or ((rows < 0) | (rows >= self.shape[0])).any()):
            raise ValueError("Expected valid physical count row indices")
        result = np.zeros((len(rows), self.shape[1]), dtype=np.int64)
        for output_row, row in enumerate(rows):
            first, last = int(self.indptr[int(row)]), int(self.indptr[int(row) + 1])
            indices, values = self.indices[first:last], self.data[first:last]
            if ((indices < 0) | (indices >= self.shape[1])).any() or (values <= 0).any() or (values >= 2**53).any():
                raise ValueError("Sparse row contains an invalid gene index/count")
            if len(indices) > 1 and (np.diff(indices) <= 0).any():
                raise ValueError("Sparse rows must have sorted unique gene columns")
            result[output_row, indices] = values
        return result


class VisiumHDV12Dataset(Dataset):
    """All eligible cached HD centers; neighbor context precedes subsampling."""

    def __init__(self, config, split="train", *, slice_index_offset=0):
        if split not in {"train", "val"}:
            raise ValueError("Only training and spatial-block validation are supported")
        self.config, self.split, self.epoch = config, split, 0
        self.slice_index_offset = _integer(slice_index_offset, "slice_index_offset")
        self.data = copy.deepcopy(config["data"])
        self.data.pop("whole_cell_max_tokens", None)
        root = Path(config["visium_cache_root"])
        self.manifest = json.loads((root / "manifest.json").read_text())
        if self.manifest.get("schema") != VISIUM_CACHE_SCHEMA or self.manifest.get("status") != "validated":
            raise ValueError("Expected a completed sparse Visium HD cache")
        if self.manifest.get("data_config") != self.data:
            raise ValueError("Visium cache data settings differ from the Xenium/V12 contract")
        vocab_path = vocabulary_path(config["assets_dir"])
        if file_sha256(vocab_path) != self.manifest.get("vocab_sha256"):
            raise ValueError("Visium cache vocabulary differs from the checkpoint vocabulary")
        vocab = json.loads(vocab_path.read_text())
        self.pad_id, self.cls_id = int(vocab["<pad>"]), int(vocab["<cls>"])
        options = config.get("v12", {})
        self.neighbors = _integer(options.get("neighbors", 8), "v12.neighbors")
        self.max_tokens = _integer(options.get("max_tokens", 512), "v12.max_tokens", 2)
        self.query_genes = _integer(options.get("query_genes", 64), "v12.query_genes", 1)
        self.gene_mask_ratio = float(options.get("gene_mask_ratio", .15))
        self.neighbor_dropout = float(options.get("neighbor_dropout", .2))
        self.max_neighbor_distance = options.get("max_neighbor_distance", 100.)
        if self.max_neighbor_distance is not None:
            self.max_neighbor_distance = float(self.max_neighbor_distance)
            if not np.isfinite(self.max_neighbor_distance) or self.max_neighbor_distance <= 0:
                raise ValueError("Invalid HD neighbor radius")
        if not 0 <= self.gene_mask_ratio <= 1 or not 0 <= self.neighbor_dropout <= 1:
            raise ValueError("Invalid HD augmentation probabilities")
        if config.get("binning") != {"num_bins": 51, "tie_policy": "left"}:
            raise ValueError("HD views retain deterministic-left 51-bin V12 inputs")
        self.lr_prior = load_lr_prior(config)
        self.lr_metadata = dict(self.lr_prior["metadata"],
            expression="masked_visible_whole_cell_raw_counts_full_measured_panel",
            graph="neighbor_to_neighbor_plus_center_receiver_from_neighbor_sender",
            physical_edge_mask="applied_by_model_from_relative_coordinates",
            center_visibility="both_compartments_must_be_observed_for_a_gene",
            center_query_values="excluded_before_activity_depth_and_coverage")
        self.target_metadata = dict(V12_TARGET_METADATA)
        self.center_filter_metadata = dict(
            rule="exclude_zero_total_counts_in_mapped_measured_panel", split=split,
            original_centers=0, retained_centers=0, excluded_centers=0, slices=[])
        self.slices, self.ends, self.indices_by_sample = [], [], {}
        previous_end, sample_ids, physical_ids = 0, set(), set()
        for item in self.manifest.get("slices", []):
            assert_not_crc_source(item)
            name = item["sample_id"]
            physical = item.get("physical_sample_id")
            physical_key = physical_sample_key(physical) if physical else None
            if (not name or name in sample_ids or not physical or physical_key in physical_ids
                    or item.get("species") != "Homo sapiens"):
                raise ValueError("HD samples need unique IDs/physical IDs and explicit human species")
            sample_ids.add(name)
            physical_ids.add(physical_key)
            folder = Path(item["cache_dir"])
            files = item.get("files_sha256")
            required_files = {"gene_ids.npy", "coords.npy", "total_counts_per_cell.npy",
                              "knn_indices.npy", "train_cell_mask.npy"}
            required_files.update(f"{prefix}.{suffix}.bin" for prefix in ("nuclear", "cytoplasmic")
                                  for suffix in ("data", "indices"))
            required_files.update(f"{prefix}.indptr.npy" for prefix in ("nuclear", "cytoplasmic"))
            required_files.update(f"{label}_{suffix}.npy" for label in ("train", "val")
                                  for suffix in ("centers", "nodes"))
            if not isinstance(files, dict) or not required_files.issubset(files):
                raise ValueError("HD cache has no immutable array checksums")
            for filename, digest in files.items():
                path = (folder / filename).resolve()
                if not path.is_relative_to(folder.resolve()) or file_sha256(path) != digest:
                    raise ValueError(f"Visium cache checksum mismatch: {name}/{filename}")
            genes = np.load(folder / "gene_ids.npy", mmap_mode="r")
            coords = np.load(folder / "coords.npy", mmap_mode="r")
            nodes = np.load(folder / f"{split}_nodes.npy", mmap_mode="r")
            centers = np.load(folder / f"{split}_centers.npy", mmap_mode="r")
            shape = (int(item["num_cells"]), int(item["num_genes"]))
            if (genes.shape != (shape[1],) or genes.dtype != np.int64
                    or len(np.unique(genes)) != len(genes)
                    or not np.isin(genes, list(vocab.values())).all()
                    or np.isin(genes, [self.pad_id, self.cls_id]).any()):
                raise ValueError("HD measured gene IDs are invalid")
            if coords.shape != (shape[0], 2) or not np.isfinite(coords).all():
                raise ValueError("HD coordinates must be finite physical micrometers")
            if (nodes.shape != (len(centers), int(self.data["context_cells"]))
                    or nodes.dtype != np.int32 or centers.dtype != np.int32
                    or not np.array_equal(nodes[:, 0], centers)
                    or ((nodes < -1) | (nodes >= shape[0])).any()):
                raise ValueError("HD fixed-patch row ordering is invalid")
            entry = dict(sample_id=name, slice_seed=stable_seed(name), genes=genes,
                         coords=coords, nodes=nodes,
                         nuclear=SparseRawCountRows(folder, "nuclear", shape, item["nuclear_nnz"]),
                         cyto=SparseRawCountRows(folder, "cytoplasmic", shape, item["cytoplasmic_nnz"]))
            row_totals = np.load(folder / "total_counts_per_cell.npy", mmap_mode="r")
            if row_totals.shape != (shape[0],) or not (row_totals[centers] > 0).all():
                raise ValueError("HD cache retained a zero-expression center")
            count = len(centers)
            audit = dict(item[split]["center_filter"], sample_id=name,
                         retained_patch_rows_sha256=file_sha256(folder / f"{split}_nodes.npy"))
            for field in ("original_centers", "retained_centers", "excluded_centers"):
                self.center_filter_metadata[field] += int(audit[field])
            self.center_filter_metadata["slices"].append(audit)
            self.slices.append(entry)
            self.ends.append(previous_end + count)
            self.indices_by_sample[name] = np.arange(previous_end, previous_end + count, dtype=np.int64)
            previous_end += count
        if not previous_end:
            raise ValueError("No eligible Visium HD centers in this split")

    def __len__(self):
        return self.ends[-1]

    def set_epoch(self, epoch):
        self.epoch = _integer(epoch, "epoch") if self.split == "train" else 0

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample_index = int(np.searchsorted(self.ends, index, side="right"))
        entry = self.slices[sample_index]
        local = index - (self.ends[sample_index - 1] if sample_index else 0)
        nodes = np.asarray(entry["nodes"][local], dtype=np.int64)
        coords = np.asarray(entry["coords"][np.maximum(nodes, 0)], dtype=np.float64)
        take = np.concatenate(([0], _neighbor_rows(nodes, coords, self.neighbors)))
        nodes, coords = nodes[take], coords[take]
        result = encode_v12_views(
            entry["nuclear"].dense_rows(nodes), entry["cyto"].dense_rows(nodes),
            entry["genes"], nodes, coords, pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            neighbors=self.neighbors, max_tokens=self.max_tokens, query_genes=self.query_genes,
            gene_mask_ratio=self.gene_mask_ratio, neighbor_dropout=self.neighbor_dropout,
            seed=int(self.data.get("seed", 42)), epoch=self.epoch, slice_seed=entry["slice_seed"],
            slice_index=self.slice_index_offset + sample_index, mask=self.split == "train",
            lr_pairs=self.lr_prior["pairs"], lr_weights=self.lr_prior["weights"],
            max_neighbor_distance=self.max_neighbor_distance)
        result["domain_id"] = torch.tensor(1, dtype=torch.int64)
        return result


class V12MixedDataset(Dataset):
    """Wrap immutable Xenium V12 data and independent sparse HD sample caches."""

    def __init__(self, config, split="train", *, include_visium=True):
        self.xenium = V12Dataset(config, split=split)
        self.visium = (VisiumHDV12Dataset(config, split=split,
                        slice_index_offset=len(self.xenium.slices)) if include_visium else None)
        self.xenium_indices = np.arange(len(self.xenium), dtype=np.int64)
        self.visium_indices = (np.arange(len(self.xenium), len(self), dtype=np.int64)
                               if self.visium is not None else np.empty(0, dtype=np.int64))
        self.visium_indices_by_sample = ({name: rows + len(self.xenium)
                                         for name, rows in self.visium.indices_by_sample.items()}
                                        if self.visium is not None else {})
        self.manifest = dict(schema=MIXED_DATASET_SCHEMA,
            xenium=copy.deepcopy(self.xenium.manifest),
            visium=copy.deepcopy(self.visium.manifest) if self.visium is not None else None,
            include_visium=bool(include_visium), split=split)
        self.center_filter_metadata = dict(xenium=self.xenium.center_filter_metadata,
            visium=self.visium.center_filter_metadata if self.visium is not None else None)
        self.target_metadata = copy.deepcopy(self.xenium.target_metadata)
        self.lr_metadata = copy.deepcopy(self.xenium.lr_metadata)
        if self.visium is not None:
            if self.target_metadata != self.visium.target_metadata or self.lr_metadata != self.visium.lr_metadata:
                raise ValueError("Xenium and HD target/LR contracts differ")

    def __len__(self):
        return len(self.xenium) + (len(self.visium) if self.visium is not None else 0)

    def set_epoch(self, epoch):
        self.xenium.set_epoch(epoch)
        if self.visium is not None:
            self.visium.set_epoch(epoch)

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < len(self.xenium):
            result = dict(self.xenium[index])
            result["domain_id"] = torch.tensor(0, dtype=torch.int64)
            return result
        return self.visium[index - len(self.xenium)]
