"""V10 nucleus/neighbor views predicting one unmasked whole-cell teacher.

This module reuses V2's split-isolated, immutable patch cache. Cached expression
is float32 log1p, reconstructed with the checked V5 integer-count inverse before
any summation or binning. Runtime V10 options belong under ``config['v10']``;
they do not change the cache protocol or turn neighboring cells into centers.

All views use deterministic-left scGPT bins and the V6-left gene selection
contract. Students are masked *before* their own visible-only quantiles are
computed. Teacher packing is independent of epoch and augmentation RNG.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .subcellular_v2 import V2Dataset
from .subcellular_v5 import _raw_count_vector, recover_raw_counts_v5
from .subcellular_v6_left import encode_whole_cell_v6_left


_STREAMS = ("nucleus", "neighbor", "teacher")
_PAD_VALUE = -2.0


def _integer(value, name, minimum=0):
    if (isinstance(value, (bool, np.bool_)) or int(value) != value
            or int(value) < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _integer_array(value, name):
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.integer):
        raise TypeError(f"{name} must contain integer IDs")
    return result.astype(np.int64, copy=False)


def _neighbor_rows(nodes, coords, neighbors):
    """Unique physical neighbors, nearest first, excluding every center copy."""
    if nodes.ndim != 1 or not len(nodes) or nodes[0] < 0:
        raise ValueError("nodes must have a real center at row zero")
    if coords.shape != (len(nodes), 2):
        raise ValueError("coords must have shape [patch rows, 2]")
    if not np.isfinite(coords[nodes >= 0]).all():
        raise ValueError("Real cells must have finite spatial coordinates")
    candidates = np.flatnonzero((nodes >= 0) & (nodes != nodes[0]))
    if not len(candidates) or neighbors == 0:
        return np.empty(0, dtype=np.int64)
    # Stable deduplication also prevents a repeated neighbor becoming multiple
    # attention keys. Center exclusion is by identity, not array position.
    _, first = np.unique(nodes[candidates], return_index=True)
    candidates = candidates[np.sort(first)]
    delta = coords[candidates].astype(np.float64) - coords[0].astype(np.float64)
    order = np.argsort(np.square(delta).sum(axis=1), kind="stable")
    return candidates[order[:neighbors]]


def _masked_counts(counts, ratio, rng):
    visible = counts.copy()
    if ratio:
        # A uniform measured-panel mask never consults expression magnitude or
        # teacher counts. Counts hidden here cannot affect the visible ranks.
        hidden = rng.random(len(visible)) < ratio
        visible[hidden] = 0
    return visible


def _encode_counts(nuclear, cyto, genes, *, pad_token_id, cls_token_id,
                   max_tokens, seed, slice_seed, node):
    return encode_whole_cell_v6_left(
        nuclear, cyto, genes, pad_token_id=pad_token_id,
        cls_token_id=cls_token_id, max_tokens=max_tokens,
        seed=np.random.SeedSequence([seed, slice_seed, int(node), 355]))


def _write_stream(output, prefix, encoded, row=None):
    length = len(encoded["whole_gene_ids"])
    index = slice(0, length) if row is None else (row, slice(0, length))
    output[f"{prefix}_gene_ids"][index] = encoded["whole_gene_ids"]
    output[f"{prefix}_bins"][index] = encoded["whole_input_bins"]
    output[f"{prefix}_valid"][index] = encoded["whole_token_valid_mask"]


def encode_v10_views(
    nuclear_raw, cytoplasm_raw, global_ids, nodes, coords, *,
    pad_token_id, cls_token_id, neighbors=8, max_tokens=512,
    gene_mask_ratio=.15, neighbor_dropout=.2, seed=42, epoch=0,
    slice_seed=0, slice_index=0, mask=True,
):
    """Encode one raw-count patch, including the center as row zero.

    ``nuclear_raw`` and ``cytoplasm_raw`` are [patch rows, measured genes]
    integer counts; ``global_ids`` maps the measured columns to the ordinary
    scGPT vocabulary. ``nodes`` are physical cell IDs and negative IDs are
    padding. ``coords`` contains matching x/y coordinates. This helper accepts
    evaluation arrays directly; never pass the cached log1p arrays here.

    Returns nucleus/teacher ``gene_ids,bins,valid`` of shape [max_tokens] and
    neighbor versions [neighbors,max_tokens], plus ``neighbor_present`` [K],
    ``relative_coords`` [K,2], ``cell_identity`` [2] and scalar
    ``nucleus_present``. CLS is valid for an empty nucleus, but every token of
    an absent/empty neighbor is invalid. The model must handle all-pad rows.

    ``mask=False`` disables both student augmentations. Teacher tokenization
    always equals ``encode_whole_cell_v6_left`` with the stable seed
    ``[seed,slice_seed,nodes[0],355]``; use its fields as the paired ordinary
    scGPT baseline. No labels are accepted or read.
    """
    neighbors = _integer(neighbors, "neighbors")
    max_tokens = _integer(max_tokens, "max_tokens", 2)
    seed, epoch = _integer(seed, "seed"), _integer(epoch, "epoch")
    slice_seed = _integer(slice_seed, "slice_seed")
    slice_index = _integer(slice_index, "slice_index")
    pad_token_id = _integer(pad_token_id, "pad_token_id")
    cls_token_id = _integer(cls_token_id, "cls_token_id")
    gene_mask_ratio, neighbor_dropout = float(gene_mask_ratio), float(neighbor_dropout)
    if not 0 <= gene_mask_ratio <= 1 or not 0 <= neighbor_dropout <= 1:
        raise ValueError("gene_mask_ratio and neighbor_dropout must lie in [0,1]")
    nuclear, cyto = np.asarray(nuclear_raw), np.asarray(cytoplasm_raw)
    if nuclear.ndim != 2 or nuclear.shape != cyto.shape or not all(nuclear.shape):
        raise ValueError("Expected nonempty matching [patch rows, panel genes] counts")
    nuclear = _raw_count_vector(nuclear.reshape(-1), "nuclear_raw").reshape(nuclear.shape)
    cyto = _raw_count_vector(cyto.reshape(-1), "cytoplasm_raw").reshape(cyto.shape)
    genes = _integer_array(global_ids, "global_ids")
    if (genes.shape != (nuclear.shape[1],) or len(np.unique(genes)) != len(genes)
            or (genes < 0).any()):
        raise ValueError("Each measured gene needs one unique nonnegative vocabulary ID")
    if pad_token_id == cls_token_id or np.isin(genes, [pad_token_id, cls_token_id]).any():
        raise ValueError("Measured genes and distinct PAD/CLS IDs must not overlap")
    nodes = _integer_array(nodes, "nodes")
    if nodes.shape != (len(nuclear),):
        raise ValueError("nodes must contain one physical cell ID per patch row")
    coords = np.asarray(coords, dtype=np.float64)
    selected = _neighbor_rows(nodes, coords, neighbors)
    if not ((nuclear[0] > 0) | (cyto[0] > 0)).any():
        raise ValueError("Teacher center has zero expression in the mapped measured panel")

    output = {}
    for prefix in _STREAMS:
        shape = (neighbors, max_tokens) if prefix == "neighbor" else (max_tokens,)
        output[f"{prefix}_gene_ids"] = np.full(shape, pad_token_id, dtype=np.int64)
        output[f"{prefix}_bins"] = np.full(shape, _PAD_VALUE, dtype=np.float32)
        output[f"{prefix}_valid"] = np.zeros(shape, dtype=bool)
    output.update(
        neighbor_present=np.zeros(neighbors, dtype=bool),
        relative_coords=np.zeros((neighbors, 2), dtype=np.float32),
        cell_identity=np.asarray([slice_index, nodes[0]], dtype=np.int64),
    )
    packing = dict(pad_token_id=pad_token_id, cls_token_id=cls_token_id,
                   max_tokens=max_tokens, seed=seed, slice_seed=slice_seed)
    teacher = _encode_counts(nuclear[0], cyto[0], genes, node=nodes[0], **packing)
    _write_stream(output, "teacher", teacher)

    nucleus_rng = np.random.default_rng(np.random.SeedSequence(
        [seed, epoch, slice_seed, int(nodes[0]), 1010]))
    nucleus = _masked_counts(nuclear[0], gene_mask_ratio if mask else 0., nucleus_rng)
    zero = np.zeros(len(genes), dtype=np.int64)
    _write_stream(output, "nucleus", _encode_counts(nucleus, zero, genes, node=nodes[0], **packing))
    output["nucleus_present"] = np.asarray((nucleus > 0).any(), dtype=bool)

    for slot, row in enumerate(selected):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, epoch, slice_seed, int(nodes[0]), int(nodes[row]), 1011]))
        if mask and rng.random() < neighbor_dropout:
            continue
        visible = _masked_counts(cyto[row], gene_mask_ratio if mask else 0., rng)
        if not (visible > 0).any():
            continue
        encoded = _encode_counts(zero, visible, genes, node=nodes[row], **packing)
        _write_stream(output, "neighbor", encoded, row=slot)
        output["neighbor_present"][slot] = True
        output["relative_coords"][slot] = coords[row] - coords[0]
    return output


class V10Dataset(V2Dataset):
    """Use the established 48-slice cache; no new split, labels or correction."""

    def __init__(self, config: dict, split="train"):
        cache_config = copy.copy(config)
        cache_config["data"] = dict(config["data"])
        # V5 introduced this runtime-only key; the immutable V2 data signature
        # intentionally excludes it. V10 uses v10.max_tokens instead.
        cache_config["data"].pop("whole_cell_max_tokens", None)
        super().__init__(cache_config, split=split)
        settings = config.get("v10", {})
        self.neighbors = _integer(settings.get("neighbors", 8), "v10.neighbors")
        self.max_tokens = _integer(settings.get("max_tokens", 512), "v10.max_tokens", 2)
        self.gene_mask_ratio = float(settings.get("gene_mask_ratio", .15))
        self.neighbor_dropout = float(settings.get("neighbor_dropout", .2))
        if not 0 <= self.gene_mask_ratio <= 1 or not 0 <= self.neighbor_dropout <= 1:
            raise ValueError("V10 augmentation probabilities must lie in [0,1]")

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slice_index = int(np.searchsorted(self.ends, index, side="right"))
        entry = self.slices[slice_index]
        local = index - (self.ends[slice_index - 1] if slice_index else 0)
        nodes = np.asarray(entry["nodes"][local], dtype=np.int64)
        if not len(nodes) or nodes[0] < 0:
            raise ValueError("Patch center cannot be padding")
        coords = np.asarray(entry["coords"][np.maximum(nodes, 0)], dtype=np.float64)
        # Recover only the actually needed rows, not the full 64-cell patch.
        take = np.concatenate(([0], _neighbor_rows(nodes, coords, self.neighbors)))
        nodes, coords = nodes[take], coords[take]
        nuclear = recover_raw_counts_v5(
            np.asarray(entry["nuclear"][nodes]), name="V10 nuclear cache")
        cyto = recover_raw_counts_v5(
            np.asarray(entry["cyto"][nodes]), name="V10 cytoplasmic cache")
        return encode_v10_views(
            nuclear, cyto, entry["genes"], nodes, coords,
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            neighbors=self.neighbors, max_tokens=self.max_tokens,
            gene_mask_ratio=self.gene_mask_ratio, neighbor_dropout=self.neighbor_dropout,
            seed=int(self.data.get("seed", 42)), epoch=self.epoch,
            slice_seed=entry["slice_seed"], slice_index=slice_index,
            mask=self.split == "train",
        )


def collate_v10(batch):
    """Stack items and independently trim trailing padding in the three views."""
    if not batch:
        raise ValueError("Cannot collate an empty V10 batch")
    lengths = {}
    for prefix in _STREAMS:
        positions = [np.flatnonzero(np.asarray(item[f"{prefix}_valid"]).reshape(
            -1, item[f"{prefix}_valid"].shape[-1]).any(axis=0)) for item in batch]
        lengths[prefix] = max([1] + [int(position[-1]) + 1 for position in positions if len(position)])
    output = {}
    for key in batch[0]:
        prefix = key.split("_", 1)[0]
        values = [item[key][..., :lengths[prefix]]
                  if key in {f"{prefix}_gene_ids", f"{prefix}_bins", f"{prefix}_valid"}
                  and prefix in lengths else item[key] for item in batch]
        output[key] = torch.from_numpy(np.stack(values))
    return output
