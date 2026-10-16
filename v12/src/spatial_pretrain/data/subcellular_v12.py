"""V12 paired nuclear/cytoplasmic count supervision, with visible-only LR.

These are simultaneous compartment observations, not temporal targets or RNA
velocity. scGPT receives independent deterministic-bin compartment views;
the decoder receives raw paired count targets. Both center compartments hide
every queried gene before binning, auxiliary count features, and LR activity.
Unmeasured panel genes are absent, never synthesized as measured zeros.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .subcellular_v5 import recover_raw_counts_v5
from .subcellular_v6 import compute_lr_features_v6
from .subcellular_v10 import _encode_counts, _integer, _integer_array, _neighbor_rows, _write_stream
from .subcellular_v11 import V11Dataset


_STREAMS = ("nucleus", "cytoplasm", "neighbor")
V12_TARGET_METADATA = {
    "expression": "joint_raw_nuclear_cytoplasmic_counts",
    "target_counts": "paired_raw_compartment_counts_and_their_sum",
    "query_sampling": "uniform_without_replacement_from_measured_panel",
    "query_mask": "hide_in_both_center_compartments_before_all_input_features",
    "input_bins": "deterministic_51_bins_over_visible_positive_genes_per_compartment",
    "input_raw_features": "log1p_visible_token_counts_and_visible_center_statistics",
    "unmeasured": "excluded_not_zero",
    "padding_target": 0,
    "padding_valid": False,
    "teacher": None,
    "temporal_target": False,
    "velocity": "not_identified_by_count_reconstruction",
}


def _raw_counts(values, name):
    """Validate without silently rounding integer arrays via float64."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        if (values < 0).any() or (values >= 2**53).any():
            raise ValueError(f"{name} must contain nonnegative integer counts below 2**53")
    elif np.issubdtype(values.dtype, np.floating):
        if (not np.isfinite(values).all() or (values < 0).any()
                or (values >= 2**53).any() or not np.array_equal(values, np.rint(values))):
            raise ValueError(f"{name} must contain finite nonnegative integer counts below 2**53")
    else:
        raise ValueError(f"{name} must contain finite nonnegative integer counts")
    return values.astype(np.int64, copy=False)


def encode_v12_views(
    nuclear_raw, cytoplasm_raw, global_ids, nodes, coords, *,
    pad_token_id, cls_token_id, neighbors=8, max_tokens=512, query_genes=64,
    gene_mask_ratio=.15, neighbor_dropout=.2, seed=42, epoch=0,
    slice_seed=0, slice_index=0, mask=True, lr_pairs=None, lr_weights=None,
    max_neighbor_distance=100.0,
):
    """Encode a raw-count patch; center is row zero, arrays are [cells, genes].

    Nuclear and cytoplasmic fields ``{gene_ids,bins,valid,log_counts}`` have
    shape [L]; neighbor token fields have shape [K,L]. Center statistics are
    ``log1p(sum N), log1p(sum C), log1p(positive N), log1p(positive C)`` over
    the masked, measured panel (before token-budget truncation). CLS/PAD raw
    count features are zero. Query counts [Q] are int64 with explicit validity.

    ``neighbor_lr_*`` [K,K] excludes center RNA entirely. ``center_lr_*`` [K]
    indexes receiver=center, sender=neighbor, and uses a center gene only if
    both compartments remain observed. Thus partial hidden receptor counts
    never masquerade as a full measured total. Its normalization depth also
    uses only these jointly visible genes. LR may use masked-visible genes
    beyond the token budget, but never queried/augmented-away center values.

    ``mask=False`` is for embedding extraction, disabling query hiding and
    all augmentations. Masked prediction validation must use ``mask=True``
    with fixed seed/epoch, gene_mask_ratio=0 and neighbor_dropout=0.
    """
    neighbors = _integer(neighbors, "neighbors")
    max_tokens = _integer(max_tokens, "max_tokens", 2)
    query_genes = _integer(query_genes, "query_genes", 1)
    seed, epoch = _integer(seed, "seed"), _integer(epoch, "epoch")
    slice_seed, slice_index = _integer(slice_seed, "slice_seed"), _integer(slice_index, "slice_index")
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
    nuclear, cyto = _raw_counts(nuclear, "nuclear_raw"), _raw_counts(cyto, "cytoplasm_raw")
    # The reused scGPT packer converts counts through float64. Keep their sum
    # in its exact-integer range, rather than silently changing a raw target.
    if (nuclear + cyto >= 2**53).any():
        raise ValueError("Summed whole-cell counts must remain below 2**53")
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
        if prefix != "neighbor":
            output[f"{prefix}_log_counts"] = np.zeros(shape, dtype=np.float32)
    output.update(
        neighbor_present=np.zeros(neighbors, dtype=bool),
        relative_coords=np.zeros((neighbors, 2), dtype=np.float32),
        cell_identity=np.asarray([slice_index, nodes[0]], dtype=np.int64),
        neighbor_cell_identity=np.full((neighbors, 2), -1, dtype=np.int64),
        query_gene_ids=np.full(query_genes, pad_token_id, dtype=np.int64),
        query_nucleus_counts=np.zeros(query_genes, dtype=np.int64),
        query_cytoplasm_counts=np.zeros(query_genes, dtype=np.int64),
        query_total_counts=np.zeros(query_genes, dtype=np.int64),
        query_valid_mask=np.zeros(query_genes, dtype=bool),
    )
    view_epoch = epoch if mask else 0
    query_rng = np.random.default_rng(np.random.SeedSequence(
        [seed, view_epoch, slice_seed, int(nodes[0]), 1210]))
    query = query_rng.permutation(len(genes))[:query_genes]
    output["query_gene_ids"][:len(query)] = genes[query]
    output["query_nucleus_counts"][:len(query)] = nuclear[0, query]
    output["query_cytoplasm_counts"][:len(query)] = cyto[0, query]
    output["query_total_counts"][:len(query)] = total[0, query]
    output["query_valid_mask"][:len(query)] = True

    packing = dict(pad_token_id=pad_token_id, cls_token_id=cls_token_id,
                   max_tokens=max_tokens, seed=seed, slice_seed=slice_seed)
    zero = np.zeros(len(genes), dtype=np.int64)
    center_visible, center_known = [], []
    for compartment, (prefix, counts) in enumerate((("nucleus", nuclear[0]), ("cytoplasm", cyto[0]))):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, view_epoch, slice_seed, int(nodes[0]), 1211 + compartment]))
        known = np.ones(len(genes), dtype=bool)
        if mask:
            known[query] = False
            known[rng.random(len(genes)) < gene_mask_ratio] = False
        visible = np.where(known, counts, 0)
        encoded = _encode_counts(visible, zero, genes, node=nodes[0], **packing)
        _write_stream(output, prefix, encoded)
        # Canonical panel ordering allows aligned raw values after the legacy
        # packer's (possibly noncanonical) token-budget sampling.
        selected_genes = encoded["whole_gene_ids"][1:]
        selected_columns = np.searchsorted(genes, selected_genes)
        output[f"{prefix}_log_counts"][1:1 + len(selected_genes)] = np.log1p(visible[selected_columns])
        output[f"{prefix}_present"] = np.asarray((visible > 0).any(), dtype=bool)
        center_visible.append(visible)
        center_known.append(known)
    output["center_statistics"] = np.log1p(np.asarray([
        center_visible[0].sum(dtype=np.float64), center_visible[1].sum(dtype=np.float64),
        np.count_nonzero(center_visible[0]), np.count_nonzero(center_visible[1]),
    ], dtype=np.float64)).astype(np.float32)

    # One small LR matrix yields both subgraphs. Since activity normalization
    # is row-local, adding the center cannot alter neighbor-neighbor evidence.
    lr_nodes = np.full(neighbors + 1, -1, dtype=np.int64)
    lr_counts = np.zeros((neighbors + 1, len(genes)), dtype=np.int64)
    lr_known = np.zeros_like(lr_counts, dtype=bool)
    lr_nodes[0] = nodes[0]
    lr_known[0] = center_known[0] & center_known[1]
    lr_counts[0] = np.where(lr_known[0], center_visible[0] + center_visible[1], 0)
    for slot, row in enumerate(selected):
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, view_epoch, slice_seed, int(nodes[0]), int(nodes[row]), 1213]))
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
        lr_nodes[slot + 1], lr_counts[slot + 1], lr_known[slot + 1] = nodes[row], visible, known
    lr = compute_lr_features_v6(
        lr_counts, genes, node_indices=lr_nodes, visible_mask=lr_known,
        lr_pairs=lr_pairs, lr_weights=lr_weights)
    output.update(
        neighbor_lr_bias=lr["lr_bias"][1:, 1:].copy(),
        neighbor_lr_coverage=lr["lr_coverage"][1:, 1:].copy(),
        center_lr_bias=lr["lr_bias"][0, 1:].copy(),
        center_lr_coverage=lr["lr_coverage"][0, 1:].copy(),
    )
    return output


prepare_view_v12 = encode_v12_views


def collate_v12(batch):
    """Stack samples, independently trimming each stream's token padding."""
    if not batch:
        raise ValueError("Cannot collate an empty V12 batch")
    keys = set(batch[0])
    if any(set(item) != keys for item in batch):
        raise ValueError("V12 batch items must share the same fields")
    lengths = {}
    for prefix in _STREAMS:
        positions = [np.flatnonzero(np.asarray(item[f"{prefix}_valid"]).reshape(
            -1, item[f"{prefix}_valid"].shape[-1]).any(axis=0)) for item in batch]
        lengths[prefix] = max([1] + [int(p[-1]) + 1 for p in positions if len(p)])
    output = {}
    for key in batch[0]:
        prefix = key.split("_", 1)[0]
        trim = prefix in lengths and key in {
            f"{prefix}_gene_ids", f"{prefix}_bins", f"{prefix}_valid", f"{prefix}_log_counts"}
        values = [item[key][..., :lengths[prefix]] if trim else item[key] for item in batch]
        output[key] = torch.from_numpy(np.stack(values))
    return output


def prepare_view_batch_v12(
    nuclear_counts, cytoplasmic_counts, gene_ids, center_indices, neighbors, coords, *,
    pad_token_id, cls_token_id, max_tokens=512, query_genes=64,
    gene_mask_ratio=.15, neighbor_dropout=.2, seed=42, epoch=0,
    slice_seed=0, slice_index=0, mask=True, lr_pairs=None, lr_weights=None,
    max_neighbor_distance=100.0,
):
    """Adapt [N,G] whole-slice raw counts and explicit [B,K] neighbor rows.

    ``mask=False`` extracts unmasked embeddings; it is not a held-out-query
    reconstruction evaluation. Neighbor rows refer to the requested centers,
    not implicitly to every cell in the whole slice.
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
        items.append(encode_v12_views(
            nuclear[nodes], cyto[nodes], gene_ids, nodes, coords[nodes],
            pad_token_id=pad_token_id, cls_token_id=cls_token_id,
            neighbors=neighbor_rows.shape[1], max_tokens=max_tokens, query_genes=query_genes,
            gene_mask_ratio=gene_mask_ratio, neighbor_dropout=neighbor_dropout,
            seed=seed, epoch=epoch, slice_seed=slice_seed, slice_index=slice_index,
            mask=mask, lr_pairs=lr_pairs, lr_weights=lr_weights,
            max_neighbor_distance=max_neighbor_distance))
    return collate_v12(items)


class V12Dataset(V11Dataset):
    """Reuse V11's immutable cache validation and zero-total-center audit.

    Runtime settings exclusively come from ``config['v12']``. The inherited
    cache filter is unchanged: a nuclear-only or cytoplasm-only center remains
    eligible. This class never modifies cached expression or sample splits.
    """

    def __init__(self, config: dict, split="train"):
        adapted = copy.copy(config)
        adapted["v11"] = copy.deepcopy(config.get("v12", {}))
        super().__init__(adapted, split=split)
        self.config = copy.copy(self.config)
        self.config.pop("v11", None)
        self.target_metadata = dict(V12_TARGET_METADATA)
        self.lr_metadata.update(
            expression="masked_visible_whole_cell_raw_counts_full_measured_panel",
            graph="neighbor_to_neighbor_plus_center_receiver_from_neighbor_sender",
            center_visibility="both_compartments_must_be_observed_for_a_gene",
            center_query_values="excluded_before_activity_depth_and_coverage",
        )

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
        nuclear = recover_raw_counts_v5(np.asarray(entry["nuclear"][nodes]), name="V12 nuclear cache")
        cyto = recover_raw_counts_v5(np.asarray(entry["cyto"][nodes]), name="V12 cytoplasmic cache")
        return encode_v12_views(
            nuclear, cyto, entry["genes"], nodes, coords,
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            neighbors=self.neighbors, max_tokens=self.max_tokens, query_genes=self.query_genes,
            gene_mask_ratio=self.gene_mask_ratio, neighbor_dropout=self.neighbor_dropout,
            seed=int(self.data.get("seed", 42)), epoch=self.epoch,
            slice_seed=entry["slice_seed"], slice_index=slice_index, mask=self.split == "train",
            lr_pairs=self.lr_prior["pairs"], lr_weights=self.lr_prior["weights"],
            max_neighbor_distance=self.max_neighbor_distance)
