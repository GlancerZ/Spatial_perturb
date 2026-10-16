"""V9 data: one-hop, two-view prediction of the center cell.

Each example is the center plus its K nearest neighbours (the first K+1 rows of
the prepared V2 patch, which are stored in KNN distance order over the whole
slice). Every row is one cell-level token sequence ``[CLS, nuclear tokens,
cytoplasmic tokens]`` with per-token compartment ids.

Center tasks (per example):
  * ``cyto`` (probability ``cyto_task_fraction``): the entire center cytoplasm
    is hidden; queries are uniform cytoplasmic panel genes.
  * ``gene``: a ``paired_mask_ratio`` share of panel genes is hidden in both
    compartments; queries come from the hidden genes, both compartments.
Neighbours are never fully visible: each hides ``neighbor_mask_ratio`` of its
panel genes in both compartments, so the context view cannot copy them.

Input bins are recomputed from visible counts only. Targets are the unmasked
center's full-panel bins. ``binning.tie_policy`` selects V7 deterministic-left
or V6 randomized ties for inputs and targets alike.

LR evidence is a [K+1] vector for the center as receiver: receptor activity from
the center's visible nuclear+cytoplasmic counts, ligand activity from each
neighbour's visible cytoplasmic counts, V6 formula and shrinkage.
"""
from __future__ import annotations

import copy

import numpy as np

from .subcellular_v2 import V2Dataset, collate_v2
from .subcellular_v3 import (
    encode_context_v3,
    full_cell_bins_v3,
    sample_paired_queries_v3,
    sample_uniform_queries_v3,
)
from .subcellular_v5 import recover_raw_counts_v5
from .subcellular_v6 import _lr_evidence, load_lr_prior
from .subcellular_v7 import encode_compartment_context_v7, full_cell_bins_v7
from .subcellular_v8 import tie_policy

TASK_GENE, TASK_CYTO = 0, 1
TARGET_PAD = -2


def v9_options(config):
    options = dict(neighbors=16, neighbor_mask_ratio=.5, cyto_task_fraction=.5)
    options.update(config.get("v9", {}))
    k = options["neighbors"]
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("v9.neighbors must be a positive integer")
    for key in ("neighbor_mask_ratio", "cyto_task_fraction"):
        if not 0 <= float(options[key]) <= 1:
            raise ValueError(f"v9.{key} must lie in [0, 1]")
    return options


def encode_cells_v9(nuclear_log, cyto_log, global_ids, *, policy, node_indices, seed, epoch,
                    slice_seed, context_tokens_per_compartment, pad_token_id, cls_token_id):
    """Pack already-visible rows with the selected tie policy (same layout)."""
    encoder = encode_compartment_context_v7 if policy == "left" else encode_context_v3
    return encoder(nuclear_log, cyto_log, global_ids, node_indices=node_indices, seed=seed,
                   epoch=epoch, slice_seed=slice_seed,
                   context_tokens_per_compartment=context_tokens_per_compartment,
                   pad_token_id=pad_token_id, cls_token_id=cls_token_id)


def full_bins_v9(nuclear_log, cyto_log, *, policy, node_index, seed, slice_seed):
    if policy == "left":
        return full_cell_bins_v7(nuclear_log, cyto_log)
    return full_cell_bins_v3(nuclear_log, cyto_log, node_index=node_index, seed=seed,
                             slice_seed=slice_seed)


def center_lr_v9(center_raw, center_known, neighbor_cyto_raw, neighbor_known, global_ids,
                 node_indices, lr_pairs, lr_weights):
    """[K+1] LR evidence for the center as receiver (entry 0 is always zero).

    ``center_raw`` / ``center_known``: [genes] visible nuclear+cytoplasmic counts
    and whether each gene is observed. ``neighbor_*``: [K+1, genes] with row 0
    ignored. Reuses the V6 evidence formula on a matrix whose receiver row is
    the center and whose sender rows are neighbour cytoplasm.
    """
    raw = np.array(neighbor_cyto_raw, dtype=np.int64, copy=True)
    known = np.array(neighbor_known, dtype=bool, copy=True)
    raw[0], known[0] = center_raw, center_known
    nodes = np.asarray(node_indices, dtype=np.int64)
    known[nodes < 0] = False
    raw = np.where(known, raw, 0)
    evidence = _lr_evidence(raw, known, np.asarray(global_ids, dtype=np.int64), nodes,
                            lr_pairs, lr_weights)
    return evidence["lr_bias"][0].astype(np.float32)


class V9Dataset(V2Dataset):
    def __init__(self, config: dict, split="train"):
        cache_config = copy.copy(config)
        cache_config["data"] = dict(config["data"])
        cache_config["data"].pop("whole_cell_max_tokens", None)
        super().__init__(cache_config, split=split)
        self.policy = tie_policy(config)
        self.options = v9_options(config)
        self.cells = self.options["neighbors"] + 1
        if self.cells > int(self.data["context_cells"]):
            raise ValueError("v9.neighbors + 1 cannot exceed the prepared patch size")
        prior = load_lr_prior(config)
        self.lr_pairs, self.lr_weights = prior["pairs"], prior["weights"]
        self.lr_metadata = prior["metadata"]
        self.lr_enabled = bool(prior["metadata"].get("enabled"))

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        slice_index = int(np.searchsorted(self.ends, index, side="right"))
        entry = self.slices[slice_index]
        local = index - (self.ends[slice_index - 1] if slice_index else 0)
        nodes = np.asarray(entry["nodes"][local], dtype=np.int64)[:self.cells]
        real = nodes >= 0
        if not real[0]:
            raise ValueError("Patch center cannot be padding")
        rows = np.maximum(nodes, 0)
        nuclear = np.array(entry["nuclear"][rows], dtype=np.float32)
        cyto = np.array(entry["cyto"][rows], dtype=np.float32)
        nuclear[~real], cyto[~real] = 0, 0
        genes = np.asarray(entry["genes"])
        panel = len(genes)
        seed = int(self.data.get("seed", 42))
        query_count = int(self.data["query_genes"])
        rng = np.random.default_rng(np.random.SeedSequence(
            [seed, self.epoch, entry["slice_seed"], int(nodes[0]), 931]))

        known = np.ones((len(nodes), 2, panel), dtype=bool)
        known[~real] = False
        task = TASK_CYTO if rng.random() < float(self.options["cyto_task_fraction"]) else TASK_GENE
        if task == TASK_CYTO:
            known[0, 1] = False
            selected = sample_uniform_queries_v3(panel, query_count, rng)
        else:
            hidden, selected = sample_paired_queries_v3(
                panel, query_count, self.data.get("paired_mask_ratio", .5), rng)
            known[0, :, hidden] = False
        ratio = float(self.options["neighbor_mask_ratio"])
        if ratio > 0:
            count = max(1, int(np.ceil(panel * ratio)))
            for row in np.flatnonzero(real[1:]) + 1:
                known[row, :, rng.choice(panel, count, replace=False)] = False
        visible_nuclear = np.where(known[:, 0], nuclear, 0).astype(np.float32)
        visible_cyto = np.where(known[:, 1], cyto, 0).astype(np.float32)

        output = encode_cells_v9(
            visible_nuclear, visible_cyto, genes, policy=self.policy, node_indices=nodes,
            seed=seed, epoch=self.epoch, slice_seed=entry["slice_seed"],
            context_tokens_per_compartment=self.data["context_tokens_per_compartment"],
            pad_token_id=self.pad_id, cls_token_id=self.cls_id)
        targets_full = full_bins_v9(nuclear[0], cyto[0], policy=self.policy,
                                    node_index=int(nodes[0]), seed=seed,
                                    slice_seed=entry["slice_seed"])
        qids = np.full(2 * query_count, self.pad_id, dtype=np.int64)
        qcomp = np.repeat(np.arange(2, dtype=np.int64), query_count)
        qvalid = np.zeros(2 * query_count, dtype=bool)
        targets = np.full(2 * query_count, TARGET_PAD, dtype=np.int64)
        for compartment in (0, 1):
            block = slice(compartment * query_count, compartment * query_count + len(selected))
            qids[block] = genes[selected]
            targets[block] = targets_full[compartment, selected]
            qvalid[block] = task == TASK_GENE or compartment == 1

        if self.lr_enabled:
            raw_nuclear = recover_raw_counts_v5(visible_nuclear, name="visible nuclear")
            raw_cyto = recover_raw_counts_v5(visible_cyto, name="visible cytoplasm")
            lr = center_lr_v9(raw_nuclear[0] + raw_cyto[0], known[0].any(axis=0),
                              raw_cyto, known[:, 1], genes, nodes,
                              self.lr_pairs, self.lr_weights)
        else:
            lr = np.zeros(len(nodes), dtype=np.float32)
        coords = np.array(entry["coords"][rows], dtype=np.float32)
        coords[~real] = 0
        output.update(coords=coords, lr_center=lr, query_gene_ids=qids,
                      query_compartment_ids=qcomp, query_valid_mask=qvalid,
                      target_bins=targets, task_mode=np.asarray(task, dtype=np.int64),
                      slice_index=np.asarray(slice_index, dtype=np.int64))
        return output


def collate_v9(batch, *, pad_token_id=None):
    del pad_token_id
    return collate_v2(batch)
