"""V2 data contract: visible-only bins, explicit queries, split-isolated patches.

Targets are float32 log1p of original integer counts. They are never recovered
from the historical float16 cache. Query targets never enter encode_context.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CACHE_SCHEMA = "subcellular_v2_float32_v1"


def vocabulary_path(assets_dir):
    """Accept an unambiguous official asset root or a weights folder.

    Plain scGPT and scGPT-spatial are distinct initialization sources. Refuse
    an ambiguous root instead of silently picking the other model's assets.
    """
    root = Path(assets_dir)
    candidates = [path for path in (
        root / "scGPT_human" / "vocab.json",
        root / "scGPT_spatial_v1" / "vocab.json",
        root / "vocab.json",
    ) if path.exists()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous vocabulary files under {root}: {candidates}")
    raise FileNotFoundError(f"Cannot find official vocabulary under {root}")


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(str(text).encode()).digest()[:4], "little")


def visible_quantile_bins(nuclear, cyto, rng, num_bins=51):
    """scGPT randomized quantile ties, without any float16 quantization."""
    values = np.concatenate((nuclear, cyto)).astype(np.float32, copy=False)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Visible log counts must be finite and nonnegative")
    result = np.zeros(values.shape, dtype=np.int64)
    positive = values > 0
    if positive.any():
        positive_values = values[positive]
        edges = np.quantile(positive_values, np.linspace(0, 1, num_bins - 1))
        left = np.digitize(positive_values, edges)
        right = np.digitize(positive_values, edges, right=True)
        result[positive] = np.ceil(
            rng.random(len(positive_values)) * (right - left) + left
        ).astype(np.int64)
    boundary = len(nuclear)
    return result[:boundary], result[boundary:]


def encode_context(
    nuclear_log, cyto_log, global_ids, *, node_indices=None, seed=42, epoch=0,
    slice_seed=0, context_tokens_per_compartment=256, pad_token_id,
    cls_token_id, masked_genes=None, masked_compartment=None,
    mask_whole_cell=False,
):
    """Encode patch rows; optional masking applies only to the first row.

    ``global_ids`` maps input columns to official vocabulary IDs. Unmatched
    panel genes must already have been explicitly excluded by the caller.
    ``masked_genes`` are *column indices*, hidden in both center compartments.
    Stable original ``node_indices`` make overlapping-patch cells identical in
    unmasked inference. Negative node IDs are padding cells. Every row has a
    valid CLS sentinel; spatial modules must also exclude negative node IDs.
    """
    nuclear_log = np.asarray(nuclear_log, dtype=np.float32)
    cyto_log = np.asarray(cyto_log, dtype=np.float32)
    global_ids = np.asarray(global_ids, dtype=np.int64)
    if nuclear_log.ndim != 2 or nuclear_log.shape != cyto_log.shape:
        raise ValueError("Expected matching [cells, panel_genes] compartment arrays")
    cells, genes = nuclear_log.shape
    if global_ids.shape != (genes,) or len(np.unique(global_ids)) != genes:
        raise ValueError("Each measured panel column needs a unique vocabulary ID")
    if context_tokens_per_compartment < 1:
        raise ValueError("Token budget must be positive")
    node_indices = (np.arange(cells, dtype=np.int64) if node_indices is None
                    else np.asarray(node_indices, dtype=np.int64))
    if node_indices.shape != (cells,):
        raise ValueError("node_indices must contain one original cell ID per row")
    if masked_compartment not in (None, 0, 1):
        raise ValueError("masked_compartment must be nuclear=0 or cytoplasmic=1")
    length = 1 + 2 * context_tokens_per_compartment
    encoded = {
        "gene_ids": np.full((cells, length), pad_token_id, dtype=np.int64),
        "input_bins": np.full((cells, length), -2., dtype=np.float32),
        "compartment_ids": np.full((cells, length), 2, dtype=np.int64),
        "token_valid_mask": np.zeros((cells, length), dtype=bool),
        "node_indices": node_indices.copy(),
    }
    encoded["gene_ids"][:, 0] = cls_token_id
    encoded["token_valid_mask"][:, 0] = True
    for row, node in enumerate(node_indices):
        if node < 0:
            continue
        # No seed dependence on patch order, expression, or hidden target.
        rng = np.random.default_rng(np.random.SeedSequence(
            [int(seed), int(epoch), int(slice_seed), int(node), 519]))
        visible = [nuclear_log[row].copy(), cyto_log[row].copy()]
        if row == 0:
            if masked_genes is not None:
                hidden = np.asarray(masked_genes, dtype=np.int64)
                visible[0][hidden] = 0
                visible[1][hidden] = 0
            if masked_compartment is not None:
                visible[masked_compartment][:] = 0
            if mask_whole_cell:
                visible[0][:] = 0
                visible[1][:] = 0
        # Quantiles see the full VISIBLE panel, before token subsampling.
        bins = visible_quantile_bins(visible[0], visible[1], rng)
        position = 1
        for compartment in (0, 1):
            selected = np.flatnonzero(visible[compartment] > 0)
            if len(selected) > context_tokens_per_compartment:
                selected = rng.choice(selected, context_tokens_per_compartment,
                                      replace=False)
            # Gene order is arbitrary and has no positional encoding.
            selected = rng.permutation(selected)
            end = position + len(selected)
            encoded["gene_ids"][row, position:end] = global_ids[selected]
            encoded["input_bins"][row, position:end] = bins[compartment][selected]
            encoded["compartment_ids"][row, position:end] = compartment
            encoded["token_valid_mask"][row, position:end] = True
            position = end
    return encoded


def sample_query_genes(nuclear, cyto, query_genes, rng, compartment=None):
    """Sample from this measured panel; balance observed positive/zero support.

    Support affects supervised query sampling only, never visible encoding.
    If one group is small, the remaining slots are filled without replacement.
    """
    support = ((np.asarray(nuclear) + np.asarray(cyto)) > 0 if compartment is None
               else np.asarray((nuclear, cyto)[compartment]) > 0)
    count = min(int(query_genes), len(support))
    positive, zero = np.flatnonzero(support), np.flatnonzero(~support)
    selected_positive = rng.choice(positive, min(len(positive), count // 2), replace=False)
    selected_zero = rng.choice(zero, min(len(zero), count - len(selected_positive)), replace=False)
    selected = np.concatenate((selected_positive, selected_zero))
    if len(selected) < count:
        remaining = np.setdiff1d(np.arange(len(support)), selected, assume_unique=False)
        selected = np.concatenate((selected, rng.choice(remaining, count - len(selected),
                                                       replace=False)))
    return rng.permutation(selected).astype(np.int64)


def sample_paired_queries(nuclear, cyto, query_genes, mask_ratio, rng):
    """Hide a uniform panel subset first, then sample supervised queries within it.

    The mask depends only on panel size and RNG, never expression support. The
    decoder budget does not change the fraction of expression hidden.
    """
    if not 0 < float(mask_ratio) < 1:
        raise ValueError("paired_mask_ratio must lie strictly between zero and one")
    panel_size = len(nuclear)
    if panel_size != len(cyto) or not panel_size:
        raise ValueError("Expected two matching nonempty measured panels")
    hidden_count = max(1, int(np.ceil(panel_size * float(mask_ratio))))
    hidden = rng.choice(panel_size, hidden_count, replace=False).astype(np.int64)
    local_queries = sample_query_genes(np.asarray(nuclear)[hidden], np.asarray(cyto)[hidden],
                                       query_genes, rng)
    return hidden, hidden[local_queries]


class V2Dataset(Dataset):
    """One central reconstruction example per prepared spatial patch."""

    def __init__(self, config: dict, split="train"):
        if split not in ("train", "val"):
            raise ValueError("V2 supports train and spatial-block validation only")
        self.config, self.split, self.epoch = config, split, 0
        self.data = config["data"]
        self.manifest = json.loads((Path(config["cache_root"]) / "manifest.json").read_text())
        if self.manifest["schema"] != CACHE_SCHEMA:
            raise ValueError("Incorrect V2 cache schema")
        if self.manifest["data_config"] != self.data:
            raise ValueError("Data configuration differs from prepared cache; run preparation again")
        vocab_path = vocabulary_path(config["assets_dir"])
        if hashlib.sha256(vocab_path.read_bytes()).hexdigest() != self.manifest["vocab_sha256"]:
            raise ValueError("Vocabulary differs from prepared cache")
        vocab = json.loads(vocab_path.read_text())
        self.pad_id, self.cls_id = int(vocab["<pad>"]), int(vocab["<cls>"])
        self.slices = []
        self.ends = []
        total = 0
        for item in self.manifest["slices"]:
            folder = Path(item["cache_dir"])
            entry = {"slice_seed": stable_seed(item["name"])}
            for key, filename in (
                ("nuclear", "nuclear_log1p_float32.npy"),
                ("cyto", "cyto_log1p_float32.npy"),
                ("genes", "gene_ids.npy"), ("coords", "coords.npy"),
                ("nodes", f"{split}_nodes.npy"),
            ):
                entry[key] = np.load(folder / filename, mmap_mode="r")
            total += len(entry["nodes"])
            self.ends.append(total)
            self.slices.append(entry)
        if not total:
            raise ValueError(f"No prepared {split} patches")

    def __len__(self):
        return self.ends[-1]

    def set_epoch(self, epoch):
        self.epoch = int(epoch) if self.split == "train" else 0

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
        rng = np.random.default_rng(np.random.SeedSequence([
            int(self.data.get("seed", 42)), self.epoch, entry["slice_seed"], int(nodes[0]), 811]))
        task_mode = int(rng.choice(3, p=[.7, .2, .1]))
        # Whole-cell masking is only meaningful with observed cytoplasm nearby.
        if task_mode == 2 and not (cyto[1:] > 0).any():
            task_mode = 0
        masked_compartment = int(rng.integers(2)) if task_mode == 1 else None
        if task_mode == 0:
            hidden_genes, selected = sample_paired_queries(
                nuclear[0], cyto[0], self.data["query_genes"],
                self.data.get("paired_mask_ratio", .5), rng)
        else:
            hidden_genes = None
            selected = sample_query_genes(nuclear[0], cyto[0], self.data["query_genes"],
                                          rng, masked_compartment)
        output = encode_context(
            nuclear, cyto, entry["genes"], node_indices=nodes,
            seed=self.data.get("seed", 42), epoch=self.epoch, slice_seed=entry["slice_seed"],
            context_tokens_per_compartment=self.data["context_tokens_per_compartment"],
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            masked_genes=hidden_genes,
            masked_compartment=masked_compartment, mask_whole_cell=task_mode == 2,
        )
        query_count = int(self.data["query_genes"])
        qids = np.full(2 * query_count, self.pad_id, dtype=np.int64)
        qcomp = np.repeat(np.arange(2, dtype=np.int64), query_count)
        qvalid = np.zeros(2 * query_count, dtype=bool)
        target = np.zeros(2 * query_count, dtype=np.float32)
        for compartment, raw in enumerate((nuclear, cyto)):
            block = slice(compartment * query_count, compartment * query_count + len(selected))
            qids[block] = entry["genes"][selected]
            target[block] = raw[0, selected]
            qvalid[block] = task_mode != 1 or compartment == masked_compartment
        coords = np.array(entry["coords"][rows], dtype=np.float32)
        coords[~real] = 0
        visible_cyto = (output["token_valid_mask"] & (output["compartment_ids"] == 1)).any(axis=1)
        has_neighbor = bool((visible_cyto & real & (nodes != nodes[0])).any())
        output.update(
            coords=coords, query_gene_ids=qids, query_compartment_ids=qcomp,
            query_valid_mask=qvalid, target_log_counts=target, target_nonzero=target > 0,
            neighbor_query_mask=qvalid & (qcomp == 1) & has_neighbor,
            task_mode=np.asarray(task_mode, dtype=np.int64),
            slice_index=np.asarray(slice_index, dtype=np.int64),
        )
        return output


def collate_v2(batch):
    """Trim trailing token padding, keeping all queries and every patch row."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    length = max(int(np.flatnonzero(item["token_valid_mask"].any(axis=0))[-1]) + 1
                 for item in batch)
    token_keys = {"gene_ids", "input_bins", "compartment_ids", "token_valid_mask"}
    output = {}
    for key in batch[0]:
        values = [item[key][:, :length] if key in token_keys else item[key] for item in batch]
        output[key] = torch.from_numpy(np.stack(values))
    return output
