"""V11 complementary nuclear and spatial inputs, with expression supervision.

The immutable V2 cache contains float32 log1p *raw* compartment counts. Recover
checked integers before adding nucleus and cytoplasm. There is no whole-cell
teacher input. Targets are deterministic 51-bin relative expression (zero is
an observed zero), not absolute counts. Unmeasured genes never become targets.

LR observes only visible, merged whole-cell RNA of genuine neighbors. Its
matrix is [receiver, sender]; physical edge eligibility is applied by the
model from relative coordinates, not inferred from a shared center's kNN set.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch

from .subcellular_v2 import V2Dataset
from .subcellular_v5 import _raw_count_vector, recover_raw_counts_v5
from .subcellular_v6 import compute_lr_features_v6, load_lr_prior
from .subcellular_v10 import (
    _encode_counts, _integer, _integer_array, _neighbor_rows, _write_stream,
)


_STREAMS = ("nucleus", "neighbor")
V11_TARGET_METADATA = {
    "expression": "relative_expression_deterministic_51_bins",
    "target_counts": "raw_nuclear_plus_raw_cytoplasmic_center",
    "zero_bin": 0,
    "positive_bins": [1, 50],
    "tie_rule": "numpy_digitize_right_false",
    "quantile_edges": "50_quantiles_over_all_positive_measured_center_genes",
    "query_sampling": "uniform_without_replacement_from_measured_panel",
    "unmeasured": "excluded_not_zero",
    "padding_target": -100,
    "teacher": None,
}


def _full_bins(counts):
    bins = np.zeros(len(counts), dtype=np.int64)
    positive = counts > 0
    if positive.any():
        values = counts[positive]
        bins[positive] = np.digitize(values, np.quantile(values, np.linspace(0, 1, 50)))
    return bins


def encode_v11_views(
    nuclear_raw, cytoplasm_raw, global_ids, nodes, coords, *,
    pad_token_id, cls_token_id, neighbors=8, max_tokens=512, query_genes=64,
    gene_mask_ratio=.15, neighbor_dropout=.2, seed=42, epoch=0,
    slice_seed=0, slice_index=0, mask=True, lr_pairs=None, lr_weights=None,
    max_neighbor_distance=100.0,
):
    """Pack one raw-count patch with the center at row zero.

    Arrays are [patch rows, measured panel genes]; ``nodes`` contains physical
    IDs, with negative IDs for padding. Duplicate centers/neighbors are removed
    by identity before nearest-neighbor selection. Gene columns may be in any
    order; canonicalization makes masking/truncation invariant to that order.

    Query genes are sampled independently of expression. During training every
    query is hidden from the nuclear input *before* visible quantiles; other
    nuclear and neighbor genes may additionally be hidden at gene_mask_ratio.
    Neighbor copies of query genes remain legal spatial evidence. Validation
    disables all masking/dropout but still returns deterministic query targets.

    Returns nucleus_{gene_ids,bins,valid} [L], neighbor versions [K,L],
    nucleus_present scalar, neighbor_present [K], relative_coords [K,2],
    cell_identity [2], neighbor_cell_identity [K,2] (padding -1),
    neighbor_lr_{bias,coverage} [K,K], and
    query_{gene_ids,bins,valid_mask} [Q]. No teacher tokens or cached features.
    query_bins=-100 is padding/unavailable, while bin=0 with valid=True is a
    genuine measured zero. All genes in global_ids are measured; no absent
    vocabulary gene is synthesized into a query.
    """
    neighbors = _integer(neighbors, "neighbors")
    max_tokens = _integer(max_tokens, "max_tokens", 2)
    query_genes = _integer(query_genes, "query_genes", 1)
    seed, epoch = _integer(seed, "seed"), _integer(epoch, "epoch")
    slice_seed = _integer(slice_seed, "slice_seed")
    slice_index = _integer(slice_index, "slice_index")
    pad_token_id = _integer(pad_token_id, "pad_token_id")
    cls_token_id = _integer(cls_token_id, "cls_token_id")
    gene_mask_ratio, neighbor_dropout = float(gene_mask_ratio), float(neighbor_dropout)
    if not 0 <= gene_mask_ratio <= 1 or not 0 <= neighbor_dropout <= 1:
        raise ValueError("gene_mask_ratio and neighbor_dropout must lie in [0,1]")
    if max_neighbor_distance is not None:
        max_neighbor_distance = float(max_neighbor_distance)
        if not np.isfinite(max_neighbor_distance) or max_neighbor_distance <= 0:
            raise ValueError("max_neighbor_distance must be positive micrometers or None")
    nuclear, cyto = np.asarray(nuclear_raw), np.asarray(cytoplasm_raw)
    if nuclear.ndim != 2 or nuclear.shape != cyto.shape or not all(nuclear.shape):
        raise ValueError("Expected nonempty matching [patch rows, panel genes] counts")
    nuclear = _raw_count_vector(nuclear.reshape(-1), "nuclear_raw").reshape(nuclear.shape)
    cyto = _raw_count_vector(cyto.reshape(-1), "cytoplasm_raw").reshape(cyto.shape)
    if (nuclear > np.iinfo(np.int64).max - cyto).any():
        raise ValueError("Summed whole-cell counts exceed int64")
    genes = _integer_array(global_ids, "global_ids")
    if (genes.shape != (nuclear.shape[1],) or len(np.unique(genes)) != len(genes)
            or (genes < 0).any()):
        raise ValueError("Each measured gene needs one unique nonnegative vocabulary ID")
    if pad_token_id == cls_token_id or np.isin(genes, [pad_token_id, cls_token_id]).any():
        raise ValueError("Measured genes and distinct PAD/CLS IDs must not overlap")
    order = np.argsort(genes, kind="stable")
    genes, nuclear, cyto = genes[order], nuclear[:, order], cyto[:, order]
    nodes = _integer_array(nodes, "nodes")
    if nodes.shape != (len(nuclear),):
        raise ValueError("nodes must contain one physical cell ID per patch row")
    coords = np.asarray(coords, dtype=np.float64)
    selected = _neighbor_rows(nodes, coords, neighbors)
    if max_neighbor_distance is not None:
        distance = np.linalg.norm(coords[selected] - coords[0], axis=1)
        selected = selected[distance <= max_neighbor_distance]
    total = nuclear + cyto
    if not (total[0] > 0).any():
        raise ValueError("Center has zero expression in the mapped measured panel")

    output = {}
    for prefix in _STREAMS:
        shape = (neighbors, max_tokens) if prefix == "neighbor" else (max_tokens,)
        output[f"{prefix}_gene_ids"] = np.full(shape, pad_token_id, dtype=np.int64)
        output[f"{prefix}_bins"] = np.full(shape, -2., dtype=np.float32)
        output[f"{prefix}_valid"] = np.zeros(shape, dtype=bool)
    output.update(
        neighbor_present=np.zeros(neighbors, dtype=bool),
        relative_coords=np.zeros((neighbors, 2), dtype=np.float32),
        cell_identity=np.asarray([slice_index, nodes[0]], dtype=np.int64),
        neighbor_cell_identity=np.full((neighbors, 2), -1, dtype=np.int64),
        query_gene_ids=np.full(query_genes, pad_token_id, dtype=np.int64),
        query_bins=np.full(query_genes, -100, dtype=np.int64),
        query_valid_mask=np.zeros(query_genes, dtype=bool),
    )
    view_epoch = epoch if mask else 0
    query_rng = np.random.default_rng(np.random.SeedSequence(
        [seed, view_epoch, slice_seed, int(nodes[0]), 1110]))
    query = query_rng.permutation(len(genes))[:query_genes]
    output["query_gene_ids"][:len(query)] = genes[query]
    output["query_bins"][:len(query)] = _full_bins(total[0])[query]
    output["query_valid_mask"][:len(query)] = True

    packing = dict(pad_token_id=pad_token_id, cls_token_id=cls_token_id,
                   max_tokens=max_tokens, seed=seed, slice_seed=slice_seed)
    nucleus_rng = np.random.default_rng(np.random.SeedSequence(
        [seed, view_epoch, slice_seed, int(nodes[0]), 1111]))
    visible_nucleus = nuclear[0].copy()
    if mask:
        visible_nucleus[query] = 0
        visible_nucleus[nucleus_rng.random(len(genes)) < gene_mask_ratio] = 0
    zero = np.zeros(len(genes), dtype=np.int64)
    _write_stream(output, "nucleus", _encode_counts(
        visible_nucleus, zero, genes, node=nodes[0], **packing))
    output["nucleus_present"] = np.asarray((visible_nucleus > 0).any(), dtype=bool)

    lr_nodes = np.full(neighbors, -1, dtype=np.int64)
    lr_counts = np.zeros((neighbors, len(genes)), dtype=np.int64)
    lr_known = np.zeros_like(lr_counts, dtype=bool)
    for slot, row in enumerate(selected):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, view_epoch, slice_seed, int(nodes[0]), int(nodes[row]), 1112]))
        if mask and rng.random() < neighbor_dropout:
            continue
        known = np.ones(len(genes), dtype=bool)
        if mask:
            known[rng.random(len(genes)) < gene_mask_ratio] = False
        visible = np.where(known, total[row], 0)
        if not (visible > 0).any():
            continue
        _write_stream(output, "neighbor", _encode_counts(
            zero, visible, genes, node=nodes[row], **packing), row=slot)
        output["neighbor_present"][slot] = True
        output["relative_coords"][slot] = coords[row] - coords[0]
        output["neighbor_cell_identity"][slot] = [slice_index, nodes[row]]
        lr_nodes[slot], lr_counts[slot], lr_known[slot] = nodes[row], visible, known
    if neighbors:
        lr = compute_lr_features_v6(
            lr_counts, genes, node_indices=lr_nodes, visible_mask=lr_known,
            lr_pairs=lr_pairs, lr_weights=lr_weights)
    else:
        lr = {"lr_bias": np.zeros((0, 0), dtype=np.float32),
              "lr_coverage": np.zeros((0, 0), dtype=np.float32)}
    output.update(neighbor_lr_bias=lr["lr_bias"], neighbor_lr_coverage=lr["lr_coverage"])
    return output


def collate_v11(batch):
    """Stack samples and independently remove trailing token padding."""
    if not batch:
        raise ValueError("Cannot collate an empty V11 batch")
    lengths = {}
    for prefix in _STREAMS:
        positions = [np.flatnonzero(np.asarray(item[f"{prefix}_valid"]).reshape(
            -1, item[f"{prefix}_valid"].shape[-1]).any(axis=0)) for item in batch]
        lengths[prefix] = max([1] + [int(p[-1]) + 1 for p in positions if len(p)])
    output = {}
    for key in batch[0]:
        prefix = key.split("_", 1)[0]
        trim = prefix in lengths and key in {
            f"{prefix}_gene_ids", f"{prefix}_bins", f"{prefix}_valid"}
        values = [item[key][..., :lengths[prefix]] if trim else item[key] for item in batch]
        output[key] = torch.from_numpy(np.stack(values))
    return output


def prepare_view_batch_v11(
    nuclear_counts, cytoplasmic_counts, gene_ids, center_indices, neighbors, coords, *,
    pad_token_id, cls_token_id, max_tokens=512, query_genes=64,
    gene_mask_ratio=.15, neighbor_dropout=.2, seed=42, epoch=0,
    slice_seed=0, slice_index=0, mask=True, lr_pairs=None, lr_weights=None,
    max_neighbor_distance=100.0,
):
    """Build a torch batch from whole-slice raw counts and explicit center rows.

    Counts are [N,G], coords [N,2], center_indices [B], neighbors [B,K]
    containing whole-slice row indices (-1 padding). Neighbor rows must already
    correspond to the B requested centers; this API never guesses whether a
    neighbor matrix refers to the full slice or a selected cohort.
    """
    nuclear, cyto = np.asarray(nuclear_counts), np.asarray(cytoplasmic_counts)
    centers = _integer_array(center_indices, "center_indices")
    neighbor_rows = _integer_array(neighbors, "neighbors")
    coords = np.asarray(coords)
    if nuclear.ndim != 2 or nuclear.shape != cyto.shape:
        raise ValueError("Counts must have matching [N,G] shapes")
    if centers.ndim != 1 or not len(centers) or ((centers < 0) | (centers >= len(nuclear))).any():
        raise ValueError("center_indices must be nonempty valid whole-slice row indices")
    if neighbor_rows.ndim != 2 or len(neighbor_rows) != len(centers):
        raise ValueError("neighbors must have shape [number of selected centers,K]")
    if (neighbor_rows >= len(nuclear)).any() or coords.shape != (len(nuclear), 2):
        raise ValueError("Invalid neighbor row indices or coordinate shape")
    items = []
    for center, nearby in zip(centers, neighbor_rows):
        nodes = np.concatenate(([center], nearby[nearby >= 0]))
        items.append(encode_v11_views(
            nuclear[nodes], cyto[nodes], gene_ids, nodes, coords[nodes],
            pad_token_id=pad_token_id, cls_token_id=cls_token_id,
            neighbors=neighbor_rows.shape[1], max_tokens=max_tokens, query_genes=query_genes,
            gene_mask_ratio=gene_mask_ratio, neighbor_dropout=neighbor_dropout,
            seed=seed, epoch=epoch, slice_seed=slice_seed, slice_index=slice_index,
            mask=mask, lr_pairs=lr_pairs, lr_weights=lr_weights,
            max_neighbor_distance=max_neighbor_distance))
    return collate_v11(items)


class V11Dataset(V2Dataset):
    """All usable centers in the existing cache, with auditable empty filtering."""

    def __init__(self, config: dict, split="train"):
        cache_config = copy.copy(config)
        cache_config["data"] = dict(config["data"])
        cache_config["data"].pop("whole_cell_max_tokens", None)
        super().__init__(cache_config, split=split)
        settings = config.get("v11", {})
        self.neighbors = _integer(settings.get("neighbors", 8), "v11.neighbors")
        self.max_tokens = _integer(settings.get("max_tokens", 512), "v11.max_tokens", 2)
        self.query_genes = _integer(settings.get("query_genes", 64), "v11.query_genes", 1)
        self.gene_mask_ratio = float(settings.get("gene_mask_ratio", .15))
        self.neighbor_dropout = float(settings.get("neighbor_dropout", .2))
        self.max_neighbor_distance = settings.get("max_neighbor_distance", 100.0)
        if self.max_neighbor_distance is not None:
            self.max_neighbor_distance = float(self.max_neighbor_distance)
            if not np.isfinite(self.max_neighbor_distance) or self.max_neighbor_distance <= 0:
                raise ValueError("v11.max_neighbor_distance must be positive micrometers or None")
        if not 0 <= self.gene_mask_ratio <= 1 or not 0 <= self.neighbor_dropout <= 1:
            raise ValueError("V11 augmentation probabilities must lie in [0,1]")
        self.lr_prior = load_lr_prior(config)
        self.lr_metadata = dict(self.lr_prior["metadata"],
            expression="visible_whole_cell_raw_counts_full_measured_panel",
            graph="neighbor_to_neighbor_center_excluded",
            physical_edge_mask="applied_by_model_from_relative_coordinates")
        self.target_metadata = dict(V11_TARGET_METADATA)
        self.center_filter_metadata = {
            "rule": "exclude_zero_total_counts_in_mapped_measured_panel",
            "split": split, "original_centers": 0, "retained_centers": 0,
            "excluded_centers": 0, "slices": [],
        }
        self.ends = []
        retained_total = 0
        for slice_index, entry in enumerate(self.slices):
            center_ids = np.asarray(entry["nodes"][:, 0], dtype=np.int64)
            if ((center_ids < 0) | (center_ids >= len(entry["nuclear"]))).any():
                raise ValueError("Prepared patch has an invalid center index")
            keep = np.zeros(len(center_ids), dtype=bool)
            for start in range(0, len(center_ids), 4096):
                rows = center_ids[start:start + 4096]
                keep[start:start + len(rows)] = (
                    (np.asarray(entry["nuclear"][rows]) > 0).any(axis=1)
                    | (np.asarray(entry["cyto"][rows]) > 0).any(axis=1))
            entry["v11_center_rows"] = np.flatnonzero(keep)
            removed = center_ids[~keep]
            audit = {"slice_index": slice_index, "slice_seed": entry["slice_seed"],
                     "original_centers": len(center_ids), "retained_centers": int(keep.sum()),
                     "excluded_centers": len(removed), "excluded_cell_ids": removed.tolist(),
                     "retained_patch_rows_sha256": hashlib.sha256(
                         entry["v11_center_rows"].astype("<i8").tobytes()).hexdigest()}
            self.center_filter_metadata["slices"].append(audit)
            for key in ("original_centers", "retained_centers", "excluded_centers"):
                self.center_filter_metadata[key] += audit[key]
            retained_total += int(keep.sum())
            self.ends.append(retained_total)
        if not retained_total:
            raise ValueError("No nonempty V11 centers remain after measured-panel filtering")

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slice_index = int(np.searchsorted(self.ends, index, side="right"))
        entry = self.slices[slice_index]
        local = index - (self.ends[slice_index - 1] if slice_index else 0)
        patch_row = int(entry["v11_center_rows"][local])
        nodes = np.asarray(entry["nodes"][patch_row], dtype=np.int64)
        coords = np.asarray(entry["coords"][np.maximum(nodes, 0)], dtype=np.float64)
        take = np.concatenate(([0], _neighbor_rows(nodes, coords, self.neighbors)))
        nodes, coords = nodes[take], coords[take]
        nuclear = recover_raw_counts_v5(
            np.asarray(entry["nuclear"][nodes]), name="V11 nuclear cache")
        cyto = recover_raw_counts_v5(
            np.asarray(entry["cyto"][nodes]), name="V11 cytoplasmic cache")
        return encode_v11_views(
            nuclear, cyto, entry["genes"], nodes, coords,
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            neighbors=self.neighbors, max_tokens=self.max_tokens, query_genes=self.query_genes,
            gene_mask_ratio=self.gene_mask_ratio, neighbor_dropout=self.neighbor_dropout,
            seed=int(self.data.get("seed", 42)), epoch=self.epoch,
            slice_seed=entry["slice_seed"], slice_index=slice_index, mask=self.split == "train",
            lr_pairs=self.lr_prior["pairs"], lr_weights=self.lr_prior["weights"],
            max_neighbor_distance=self.max_neighbor_distance)
