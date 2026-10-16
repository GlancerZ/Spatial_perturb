"""Shared full-cell bins for dual-compartment CLS reconstruction.

V3 reuses the validated V2 float32 log1p cache. A cell's complete nuclear and
cytoplasmic panel defines one scGPT-style quantile table, before any masking.
This is denoising in a pre-binned space: hidden counts can affect the ranks of
visible counts. It is deliberately *not* raw-count visible-only isolation.
Target tensors never leave the dataset through ``encode_context_v3``.

Zero means an observed count of zero; positive bins are 1..50. PAD is -2 and
masked genes are removed rather than represented by a zero or MASK token.
"""
from __future__ import annotations

import numpy as np

from .subcellular_v2 import V2Dataset, collate_v2, visible_quantile_bins


def full_cell_bins_v3(nuclear_log, cyto_log, *, node_index, seed=42, slice_seed=0):
    """Return [2, panel_genes] bins, independent of mask, patch order and epoch.

    log1p is monotone, so binning the cached float32 values uses the original
    count ordering at cache precision without reconstructing absolute counts.
    A dedicated per-cell RNG keeps randomized quantile ties independent of
    token subsampling, query sampling and task selection.
    """
    nuclear = np.asarray(nuclear_log, dtype=np.float32)
    cyto = np.asarray(cyto_log, dtype=np.float32)
    if nuclear.ndim != 1 or nuclear.shape != cyto.shape or not nuclear.size:
        raise ValueError("Expected two matching nonempty measured panels")
    if int(node_index) < 0:
        raise ValueError("Full-cell bins require a real nonnegative cell ID")
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed), int(slice_seed), int(node_index), 351]))
    return np.stack(visible_quantile_bins(nuclear, cyto, rng, num_bins=51))


def sample_uniform_queries_v3(panel_size, query_genes, rng, *, candidates=None):
    """Uniform measured-panel sampling; expression support is not an input."""
    panel_size, query_genes = int(panel_size), int(query_genes)
    if panel_size < 1 or query_genes < 1:
        raise ValueError("Panel size and query budget must be positive")
    eligible = (np.arange(panel_size, dtype=np.int64) if candidates is None
                else np.asarray(candidates, dtype=np.int64))
    if (eligible.ndim != 1 or not eligible.size
            or (eligible < 0).any() or (eligible >= panel_size).any()
            or len(np.unique(eligible)) != len(eligible)):
        raise ValueError("Query candidates must be unique measured-panel indices")
    return rng.choice(eligible, min(query_genes, len(eligible)), replace=False).astype(np.int64)


def sample_paired_queries_v3(panel_size, query_genes, mask_ratio, rng):
    """Choose a uniform panel mask, then uniform queries within that mask."""
    if int(panel_size) < 1 or not 0 < float(mask_ratio) < 1:
        raise ValueError("A nonempty panel and paired_mask_ratio strictly inside (0, 1) are required")
    hidden_count = max(1, int(np.ceil(int(panel_size) * float(mask_ratio))))
    hidden = rng.choice(int(panel_size), hidden_count, replace=False).astype(np.int64)
    queries = sample_uniform_queries_v3(panel_size, query_genes, rng, candidates=hidden)
    return hidden, queries


def encode_context_v3(
    nuclear_log, cyto_log, global_ids, *, node_indices=None, seed=42, epoch=0,
    slice_seed=0, context_tokens_per_compartment=256, pad_token_id,
    cls_token_id, masked_genes=None, masked_compartment=None,
    mask_whole_cell=False,
):
    """Pack visible tokens; only the first row receives the requested mask.

    Return only encoder fields, in the V2 packed layout. The V3 model separates
    nuclear/cytoplasmic tokens and adds each sequence's original scGPT CLS.
    Full-panel bins are computed before masking and shared with query targets;
    changing hidden raw values may change visible quantile ranks by design.
    ``epoch`` affects token order/subsampling, never the underlying bin table.
    Every real or padded row keeps one valid non-gene CLS sentinel.
    """
    nuclear = np.asarray(nuclear_log, dtype=np.float32)
    cyto = np.asarray(cyto_log, dtype=np.float32)
    global_ids = np.asarray(global_ids, dtype=np.int64)
    if nuclear.ndim != 2 or nuclear.shape != cyto.shape:
        raise ValueError("Expected matching [cells, panel_genes] compartment arrays")
    cells, genes = nuclear.shape
    if not cells or not genes:
        raise ValueError("At least one patch row and one measured panel gene are required")
    if global_ids.shape != (genes,) or len(np.unique(global_ids)) != genes:
        raise ValueError("Each measured panel column needs a unique vocabulary ID")
    if context_tokens_per_compartment < 1:
        raise ValueError("Token budget must be positive")
    if (not np.isfinite(nuclear).all() or not np.isfinite(cyto).all()
            or (nuclear < 0).any() or (cyto < 0).any()):
        raise ValueError("Cached log counts must be finite and nonnegative")
    nodes = (np.arange(cells, dtype=np.int64) if node_indices is None
             else np.asarray(node_indices, dtype=np.int64))
    if nodes.shape != (cells,):
        raise ValueError("node_indices must contain one original cell ID per row")
    if masked_compartment not in (None, 0, 1):
        raise ValueError("masked_compartment must be nuclear=0 or cytoplasmic=1")
    hidden = None if masked_genes is None else np.asarray(masked_genes, dtype=np.int64)
    if hidden is not None and (hidden.ndim != 1 or (hidden < 0).any() or (hidden >= genes).any()):
        raise ValueError("masked_genes must contain measured-panel column indices")
    length = 1 + 2 * int(context_tokens_per_compartment)
    encoded = {
        "gene_ids": np.full((cells, length), pad_token_id, dtype=np.int64),
        "input_bins": np.full((cells, length), -2., dtype=np.float32),
        "compartment_ids": np.full((cells, length), 2, dtype=np.int64),
        "token_valid_mask": np.zeros((cells, length), dtype=bool),
        "node_indices": nodes.copy(),
    }
    encoded["gene_ids"][:, 0] = cls_token_id
    encoded["token_valid_mask"][:, 0] = True
    for row, node in enumerate(nodes):
        if node < 0:
            continue
        full_bins = full_cell_bins_v3(nuclear[row], cyto[row], node_index=node,
                                     seed=seed, slice_seed=slice_seed)
        visible = np.stack((nuclear[row] > 0, cyto[row] > 0))
        if row == 0:
            if hidden is not None:
                visible[:, hidden] = False
            if masked_compartment is not None:
                visible[masked_compartment] = False
            if mask_whole_cell:
                visible[:] = False
        token_rng = np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(epoch), int(slice_seed), int(node), 353]))
        position = 1
        for compartment in (0, 1):
            selected = np.flatnonzero(visible[compartment])
            if len(selected) > context_tokens_per_compartment:
                selected = token_rng.choice(selected, context_tokens_per_compartment, replace=False)
            selected = token_rng.permutation(selected)
            end = position + len(selected)
            encoded["gene_ids"][row, position:end] = global_ids[selected]
            encoded["input_bins"][row, position:end] = full_bins[compartment, selected]
            encoded["compartment_ids"][row, position:end] = compartment
            encoded["token_valid_mask"][row, position:end] = True
            position = end
    return encoded


class V3Dataset(V2Dataset):
    """Reuse V2 cache/index validation; supervise only stable relative bins."""

    def __init__(self, config: dict, split="train"):
        super().__init__(config, split=split)
        if int(self.data["query_genes"]) < 1:
            raise ValueError("query_genes must be positive")
        if not 0 < float(self.data.get("paired_mask_ratio", .5)) < 1:
            raise ValueError("paired_mask_ratio must lie strictly between zero and one")

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
        # A hidden center needs genuinely observed, non-self neighbor cytoplasm.
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
        output = encode_context_v3(
            nuclear, cyto, entry["genes"], node_indices=nodes, seed=seed,
            epoch=self.epoch, slice_seed=entry["slice_seed"],
            context_tokens_per_compartment=self.data["context_tokens_per_compartment"],
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            masked_genes=hidden_genes, masked_compartment=masked_compartment,
            mask_whole_cell=task_mode == 2,
        )
        full_bins = full_cell_bins_v3(nuclear[0], cyto[0], node_index=nodes[0],
                                     seed=seed, slice_seed=entry["slice_seed"])
        qids = np.full(2 * query_count, self.pad_id, dtype=np.int64)
        qcomp = np.repeat(np.arange(2, dtype=np.int64), query_count)
        qvalid = np.zeros(2 * query_count, dtype=bool)
        targets = np.full(2 * query_count, -2, dtype=np.int64)
        for compartment in (0, 1):
            block = slice(compartment * query_count, compartment * query_count + len(selected))
            qids[block] = entry["genes"][selected]
            targets[block] = full_bins[compartment, selected]
            qvalid[block] = task_mode != 1 or compartment == masked_compartment
        coords = np.array(entry["coords"][rows], dtype=np.float32)
        coords[~real] = 0
        visible = np.array([
            (output["token_valid_mask"][0] & (output["compartment_ids"][0] == compartment)).any()
            for compartment in (0, 1)
        ])
        output.update(
            coords=coords, query_gene_ids=qids, query_compartment_ids=qcomp,
            query_valid_mask=qvalid, target_bins=targets, target_nonzero=targets > 0,
            compartment_visible_mask=visible,
            task_mode=np.asarray(task_mode, dtype=np.int64),
            slice_index=np.asarray(slice_index, dtype=np.int64),
        )
        return output


def collate_v3(batch):
    """Trim only token PAD; preserve query masks and relative-bin targets."""
    return collate_v2(batch)
