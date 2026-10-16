"""V7 deterministic-left binning for the V6 subcellular/LR architecture.

V7 preserves V6 masking, query sampling, token subsampling and LR evidence.
Only quantile-tie handling changes: equal positive values inside one cell and
one binning domain always use NumPy's left/default ``digitize`` result.  This
is the same tie policy used by the deterministic ordinary-scGPT baseline.
"""
from __future__ import annotations

import numpy as np

from .subcellular_v3 import sample_paired_queries_v3, sample_uniform_queries_v3
from .subcellular_v5 import (
    WHOLE_PAD_VALUE,
    _raw_count_vector,
    _whole_rng,
    recover_raw_counts_v5,
)
from .subcellular_v6 import V6Dataset, _lr_evidence, _visible_view, collate_v6


TIE_POLICY = "left"
NUM_BINS = 51


def deterministic_quantile_bins_v7(nuclear, cyto, *, num_bins=NUM_BINS):
    """Return shared N/C per-cell bins without randomized tie spreading."""
    nuclear = np.asarray(nuclear, dtype=np.float32)
    cyto = np.asarray(cyto, dtype=np.float32)
    if nuclear.ndim != 1 or nuclear.shape != cyto.shape or not nuclear.size:
        raise ValueError("Expected two matching nonempty measured panels")
    if (not np.isfinite(nuclear).all() or not np.isfinite(cyto).all()
            or (nuclear < 0).any() or (cyto < 0).any()):
        raise ValueError("Counts must be finite and nonnegative")
    if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("num_bins must be an integer of at least two")
    values = np.concatenate((nuclear, cyto))
    result = np.zeros(values.shape, dtype=np.int64)
    positive = values > 0
    if positive.any():
        selected = values[positive]
        edges = np.quantile(selected, np.linspace(0, 1, num_bins - 1))
        result[positive] = np.digitize(selected, edges)
        if (result[positive] < 1).any() or (result[positive] >= num_bins).any():
            raise AssertionError("Positive deterministic bins lie outside [1, num_bins-1]")
    boundary = len(nuclear)
    return result[:boundary], result[boundary:]


def full_cell_bins_v7(nuclear_log, cyto_log):
    """Deterministic full-panel target bins; independent of masks and RNG."""
    return np.stack(deterministic_quantile_bins_v7(nuclear_log, cyto_log))


def encode_compartment_context_v7(
    nuclear_log, cyto_log, global_ids, *, node_indices=None, seed=42, epoch=0,
    slice_seed=0, context_tokens_per_compartment=256, pad_token_id,
    cls_token_id,
):
    """Pack already-visible N/C inputs using deterministic-left bin values."""
    nuclear = np.asarray(nuclear_log, dtype=np.float32)
    cyto = np.asarray(cyto_log, dtype=np.float32)
    genes = np.asarray(global_ids, dtype=np.int64)
    if nuclear.ndim != 2 or nuclear.shape != cyto.shape:
        raise ValueError("Expected matching [cells, panel_genes] compartment arrays")
    cells, panel_size = nuclear.shape
    if not cells or not panel_size:
        raise ValueError("At least one patch row and measured gene are required")
    if genes.shape != (panel_size,) or len(np.unique(genes)) != panel_size:
        raise ValueError("Each measured panel column needs a unique vocabulary ID")
    if (isinstance(context_tokens_per_compartment, bool)
            or not isinstance(context_tokens_per_compartment, int)
            or context_tokens_per_compartment < 1):
        raise ValueError("context_tokens_per_compartment must be positive")
    if (not np.isfinite(nuclear).all() or not np.isfinite(cyto).all()
            or (nuclear < 0).any() or (cyto < 0).any()):
        raise ValueError("Visible cached log counts must be finite and nonnegative")
    nodes = (np.arange(cells, dtype=np.int64) if node_indices is None
             else np.asarray(node_indices, dtype=np.int64))
    if nodes.shape != (cells,) or nodes[0] < 0:
        raise ValueError("node_indices must contain one ID per row and a real center")
    length = 1 + 2 * context_tokens_per_compartment
    output = {
        "gene_ids": np.full((cells, length), int(pad_token_id), dtype=np.int64),
        "input_bins": np.full((cells, length), WHOLE_PAD_VALUE, dtype=np.float32),
        "compartment_ids": np.full((cells, length), 2, dtype=np.int64),
        "token_valid_mask": np.zeros((cells, length), dtype=bool),
        "node_indices": nodes.copy(),
    }
    output["gene_ids"][:, 0] = int(cls_token_id)
    output["token_valid_mask"][:, 0] = True
    for row, node in enumerate(nodes):
        if node < 0:
            continue
        bins = deterministic_quantile_bins_v7(nuclear[row], cyto[row])
        token_rng = np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(epoch), int(slice_seed), int(node), 353]))
        position = 1
        for compartment in (0, 1):
            selected = np.flatnonzero((nuclear, cyto)[compartment][row] > 0)
            if len(selected) > context_tokens_per_compartment:
                selected = token_rng.choice(
                    selected, context_tokens_per_compartment, replace=False)
            selected = token_rng.permutation(selected)
            end = position + len(selected)
            output["gene_ids"][row, position:end] = genes[selected]
            output["input_bins"][row, position:end] = bins[compartment][selected]
            output["compartment_ids"][row, position:end] = compartment
            output["token_valid_mask"][row, position:end] = True
            position = end
    return output


def encode_whole_cell_v7(
    nuclear_counts, cyto_counts, global_ids, *, pad_token_id, cls_token_id,
    max_tokens=1200, rng=None, seed=None, masked_genes=None,
    masked_compartment=None, mask_whole_cell=False,
):
    """Encode visible whole-cell counts with deterministic-left tie handling."""
    nuclear = _raw_count_vector(nuclear_counts, "nuclear_counts").copy()
    cyto = _raw_count_vector(cyto_counts, "cyto_counts").copy()
    genes = np.asarray(global_ids, dtype=np.int64)
    pad_token_id, cls_token_id = int(pad_token_id), int(cls_token_id)
    if nuclear.shape != cyto.shape or genes.shape != nuclear.shape or not len(genes):
        raise ValueError("Counts and global_ids must be matching nonempty panels")
    if len(np.unique(genes)) != len(genes):
        raise ValueError("Each measured panel column needs a unique vocabulary ID")
    if pad_token_id == cls_token_id or np.isin(genes, [pad_token_id, cls_token_id]).any():
        raise ValueError("Measured genes and distinct PAD/CLS IDs must not overlap")
    if (isinstance(max_tokens, (bool, np.bool_))
            or int(max_tokens) != max_tokens or int(max_tokens) < 1):
        raise ValueError("max_tokens must be a positive integer")
    if masked_compartment not in (None, 0, 1):
        raise ValueError("masked_compartment must be nuclear=0 or cytoplasmic=1")
    hidden = None if masked_genes is None else np.asarray(masked_genes, dtype=np.int64)
    if hidden is not None:
        if (hidden.ndim != 1 or (hidden < 0).any() or (hidden >= len(genes)).any()
                or len(np.unique(hidden)) != len(hidden)):
            raise ValueError("masked_genes must be unique measured-panel indices")
        nuclear[hidden] = 0
        cyto[hidden] = 0
    if masked_compartment is not None:
        (nuclear, cyto)[masked_compartment][:] = 0
    if mask_whole_cell:
        nuclear[:] = 0
        cyto[:] = 0
    if (nuclear > np.iinfo(np.int64).max - cyto).any():
        raise ValueError("Summed whole-cell counts exceed int64")
    total = nuclear + cyto
    expressed = np.flatnonzero(total > 0)
    bins = np.empty(len(expressed), dtype=np.int64)
    if len(expressed):
        values = total[expressed]
        edges = np.quantile(values, np.linspace(0, 1, NUM_BINS - 1))
        bins[:] = np.digitize(values, edges)
        if (bins < 1).any() or (bins >= NUM_BINS).any():
            raise AssertionError("Ordinary scGPT positive bins must lie in [1, 50]")
    gene_budget = int(max_tokens) - 1
    if len(expressed) > gene_budget:
        generator = _whole_rng(rng, seed)
        selected = generator.permutation(len(expressed))[:gene_budget]
        expressed, bins = expressed[selected], bins[selected]
    output = {
        "whole_gene_ids": np.empty(1 + len(expressed), dtype=np.int64),
        "whole_input_bins": np.empty(1 + len(expressed), dtype=np.float32),
        "whole_token_valid_mask": np.ones(1 + len(expressed), dtype=bool),
    }
    output["whole_gene_ids"][0] = cls_token_id
    output["whole_input_bins"][0] = WHOLE_PAD_VALUE
    output["whole_gene_ids"][1:] = genes[expressed]
    output["whole_input_bins"][1:] = bins
    return output


def encode_context_v7(
    nuclear_log, cyto_log, global_ids, *, node_indices=None, seed=42, epoch=0,
    slice_seed=0, context_tokens_per_compartment=256, pad_token_id,
    cls_token_id, whole_cell_max_tokens=1200, masked_genes=None,
    masked_compartment=None, mask_whole_cell=False,
    whole_nuclear_counts=None, whole_cyto_counts=None, whole_rng=None,
    whole_seed=None, lr_pairs=None, lr_weights=None,
):
    """V6 visibility/LR semantics with deterministic input binning."""
    nuclear, cyto, genes, nodes, known = _visible_view(
        nuclear_log, cyto_log, global_ids, node_indices,
        masked_genes, masked_compartment, mask_whole_cell)
    output = encode_compartment_context_v7(
        nuclear, cyto, genes, node_indices=nodes, seed=seed, epoch=epoch,
        slice_seed=slice_seed,
        context_tokens_per_compartment=context_tokens_per_compartment,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id)
    if (whole_nuclear_counts is None) != (whole_cyto_counts is None):
        raise ValueError("Supply both raw whole-cell compartments or neither")
    if whole_nuclear_counts is None:
        whole_nuclear_counts = recover_raw_counts_v5(
            nuclear[0], name="visible nuclear center")
        whole_cyto_counts = recover_raw_counts_v5(
            cyto[0], name="visible cytoplasmic center")
    if whole_rng is None and whole_seed is None:
        whole_seed = np.random.SeedSequence(
            [int(seed), int(slice_seed), int(nodes[0]), 355])
    output.update(encode_whole_cell_v7(
        whole_nuclear_counts, whole_cyto_counts, genes,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id,
        max_tokens=whole_cell_max_tokens, rng=whole_rng, seed=whole_seed,
        masked_genes=masked_genes, masked_compartment=masked_compartment,
        mask_whole_cell=mask_whole_cell))
    visible_raw = (None if lr_pairs is None or np.size(lr_pairs) == 0 else
                   recover_raw_counts_v5(cyto, name="visible cytoplasmic LR counts"))
    output.update(_lr_evidence(
        visible_raw, known[:, 1], genes, nodes, lr_pairs, lr_weights))
    return output


class V7Dataset(V6Dataset):
    """V6 tasks and cache with deterministic-left inputs and targets."""

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slice_index = int(np.searchsorted(self.ends, index, side="right"))
        entry = self.slices[slice_index]
        local = index - (self.ends[slice_index - 1] if slice_index else 0)
        nodes = np.asarray(entry["nodes"][local], dtype=np.int64)
        real = nodes >= 0
        if not real[0]:
            raise ValueError("Patch center cannot be padding")
        rows = np.maximum(nodes, 0)
        nuclear = np.array(entry["nuclear"][rows], dtype=np.float32)
        cyto = np.array(entry["cyto"][rows], dtype=np.float32)
        nuclear[~real], cyto[~real] = 0, 0
        panel_size = len(entry["genes"])
        query_count = int(self.data["query_genes"])
        seed = int(self.data.get("seed", 42))
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, self.epoch, entry["slice_seed"], int(nodes[0]), 831]))
        task_mode = int(rng.choice(3, p=[.7, .2, .1]))
        neighbor_rows = real & (nodes != nodes[0])
        if task_mode == 2 and not (cyto[neighbor_rows] > 0).any():
            task_mode = 0
        masked_compartment = int(rng.integers(2)) if task_mode == 1 else None
        if task_mode == 0:
            hidden_genes, selected = sample_paired_queries_v3(
                panel_size, query_count, self.data.get("paired_mask_ratio", .5), rng)
        else:
            hidden_genes = None
            selected = sample_uniform_queries_v3(panel_size, query_count, rng)
        output = encode_context_v7(
            nuclear, cyto, entry["genes"], node_indices=nodes, seed=seed,
            epoch=self.epoch, slice_seed=entry["slice_seed"],
            context_tokens_per_compartment=self.data["context_tokens_per_compartment"],
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            whole_cell_max_tokens=self.whole_cell_max_tokens,
            masked_genes=hidden_genes, masked_compartment=masked_compartment,
            mask_whole_cell=task_mode == 2,
            lr_pairs=self.lr_pairs, lr_weights=self.lr_weights)
        full_bins = full_cell_bins_v7(nuclear[0], cyto[0])
        qids = np.full(2 * query_count, self.pad_id, dtype=np.int64)
        qcomp = np.repeat(np.arange(2, dtype=np.int64), query_count)
        qvalid = np.zeros(2 * query_count, dtype=bool)
        targets = np.full(2 * query_count, WHOLE_PAD_VALUE, dtype=np.int64)
        for compartment in (0, 1):
            block = slice(compartment * query_count,
                          compartment * query_count + len(selected))
            qids[block] = entry["genes"][selected]
            targets[block] = full_bins[compartment, selected]
            qvalid[block] = task_mode != 1 or compartment == masked_compartment
        coords = np.array(entry["coords"][rows], dtype=np.float32)
        coords[~real] = 0
        visible = np.array([
            (output["token_valid_mask"][0]
             & (output["compartment_ids"][0] == compartment)).any()
            for compartment in (0, 1)])
        output.update(
            coords=coords, query_gene_ids=qids, query_compartment_ids=qcomp,
            query_valid_mask=qvalid, target_bins=targets,
            target_nonzero=targets > 0, compartment_visible_mask=visible,
            task_mode=np.asarray(task_mode, dtype=np.int64),
            slice_index=np.asarray(slice_index, dtype=np.int64))
        return output


def collate_v7(batch, *, pad_token_id=None):
    return collate_v6(batch, pad_token_id=pad_token_id)
