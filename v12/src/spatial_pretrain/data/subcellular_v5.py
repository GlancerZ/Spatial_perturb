"""V5 data: the V3 compartment task plus a visible whole-cell CLS input.

The compartment branch intentionally retains V3's 51-bin, pre-mask target and
input table.  Consequently its visible bin ranks may still depend on hidden
counts.  The additional ordinary-scGPT branch has a narrower contract: the
same sampled center-cell mask is applied to raw nuclear/cytoplasmic counts
*before* they are summed and quantile binned.  It therefore never receives a
hidden gene, hidden compartment, or unmasked whole-cell value.

V2's cache stores float32 ``log1p`` values.  Integer counts are reconstructed
only when the cached value identifies one unique nonnegative integer; lossy
values fail rather than silently changing the ordinary whole-cell input.

V5 uses a dedicated per-cell NumPy Generator.  It implements the official
random-tie formula, but its stream is not the historical global-NumPy/torch
stream of ``DataCollator``.  A step-0 comparison therefore needs a freshly
paired ordinary-scGPT baseline built from the same V5 whole-cell tokens.
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from .subcellular_v2 import V2Dataset, collate_v2
from .subcellular_v3 import (
    encode_context_v3,
    full_cell_bins_v3,
    sample_paired_queries_v3,
    sample_uniform_queries_v3,
)


WHOLE_KEYS = ("whole_gene_ids", "whole_input_bins", "whole_token_valid_mask")
WHOLE_PAD_VALUE = -2.0


def recover_raw_counts_v5(log_counts, *, name="counts"):
    """Recover uniquely represented integer counts from the V2 float32 cache.

    Cache preparation computes ``np.log1p(raw.astype(np.float32))``.  Merely
    rounding ``expm1`` can look plausible after float32 has merged neighboring
    integers, so this function also checks the forward cache transform and
    rejects a value if either adjacent integer produces the same float32 log.
    """
    cached = np.asarray(log_counts)
    if cached.dtype != np.float32:
        raise TypeError(f"{name} must be float32 log1p cache values")
    if not np.isfinite(cached).all() or (cached < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative log1p values")

    estimate = np.expm1(cached.astype(np.float64))
    int64_max = np.iinfo(np.int64).max
    if (not np.isfinite(estimate).all() or (estimate < 0).any()
            or (estimate >= int64_max).any()):
        raise ValueError(f"{name} lies outside the recoverable integer-count range")
    recovered = np.rint(estimate).astype(np.int64)

    def cache_transform(values):
        return np.log1p(values.astype(np.float32)).astype(np.float32, copy=False)

    if not np.array_equal(cache_transform(recovered), cached):
        raise ValueError(f"{name} is not an exact float32 log1p encoding of integer counts")
    lower = np.maximum(recovered - 1, 0)
    ambiguous_lower = (recovered > 0) & (cache_transform(lower) == cached)
    ambiguous_upper = cache_transform(recovered + 1) == cached
    if ambiguous_lower.any() or ambiguous_upper.any():
        raise ValueError(
            f"{name} does not uniquely identify integer counts at float32 precision")
    return recovered


def _raw_count_vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional measured panel")
    numeric = array.astype(np.float64)
    if (not np.isfinite(numeric).all() or (numeric < 0).any()
            or (numeric > np.iinfo(np.int64).max).any()
            or not np.array_equal(numeric, np.rint(numeric))):
        raise ValueError(f"{name} must contain finite nonnegative integer counts")
    return numeric.astype(np.int64)


def _whole_rng(rng, seed):
    if rng is not None and seed is not None:
        raise ValueError("Pass either rng or seed, not both")
    if rng is None:
        rng = np.random.default_rng(42 if seed is None else seed)
    if not all(hasattr(rng, method) for method in ("random", "permutation")):
        raise TypeError("rng must provide random and permutation methods")
    return rng


def encode_whole_cell_v5(
    nuclear_counts, cyto_counts, global_ids, *, pad_token_id, cls_token_id,
    max_tokens=1200, rng=None, seed=None, masked_genes=None,
    masked_compartment=None, mask_whole_cell=False,
):
    """Encode one center cell with ordinary scGPT's 51-bin token semantics.

    Inputs are raw discrete counts, which allows evaluation to call this helper
    without a V2 cache.  The center mask is applied first, then compartments are
    summed.  Zeros are omitted.  Randomized quantile ties use ``rng.random`` in
    the same formula as scGPT ``_digitize``.  Binning covers every visible
    positive gene before an overlength sequence is sampled.  The returned item
    fields are variable-length ``[L]`` arrays and always retain a valid CLS.

    An explicit Generator is convenient for reference comparisons.  ``seed``
    accepts an integer or ``SeedSequence`` and is mutually exclusive with it.
    """
    nuclear = _raw_count_vector(nuclear_counts, "nuclear_counts").copy()
    cyto = _raw_count_vector(cyto_counts, "cyto_counts").copy()
    genes = np.asarray(global_ids, dtype=np.int64)
    pad_token_id, cls_token_id = int(pad_token_id), int(cls_token_id)
    if nuclear.shape != cyto.shape or genes.shape != nuclear.shape or not len(genes):
        raise ValueError("Counts and global_ids must be matching nonempty measured panels")
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
            raise ValueError("masked_genes must be unique measured-panel column indices")
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
    generator = _whole_rng(rng, seed)
    bins = np.empty(len(expressed), dtype=np.int64)
    if len(expressed):
        values = total[expressed]
        edges = np.quantile(values, np.linspace(0, 1, 50))
        left = np.digitize(values, edges)
        right = np.digitize(values, edges, right=True)
        bins[:] = np.ceil(
            generator.random(len(values)) * (right - left) + left
        ).astype(np.int64)
        if (bins < 1).any() or (bins > 50).any():
            raise AssertionError("Ordinary scGPT positive bins must lie in [1, 50]")

    gene_budget = int(max_tokens) - 1
    if len(expressed) > gene_budget:
        selected = generator.permutation(len(expressed))[:gene_budget]
        expressed, bins = expressed[selected], bins[selected]
    length = 1 + len(expressed)
    output = {
        "whole_gene_ids": np.empty(length, dtype=np.int64),
        "whole_input_bins": np.empty(length, dtype=np.float32),
        "whole_token_valid_mask": np.ones(length, dtype=bool),
    }
    output["whole_gene_ids"][0] = cls_token_id
    output["whole_input_bins"][0] = WHOLE_PAD_VALUE
    output["whole_gene_ids"][1:] = genes[expressed]
    output["whole_input_bins"][1:] = bins
    return output


def encode_context_v5(
    nuclear_log, cyto_log, global_ids, *, node_indices=None, seed=42, epoch=0,
    slice_seed=0, context_tokens_per_compartment=256, pad_token_id,
    cls_token_id, whole_cell_max_tokens=1200, masked_genes=None,
    masked_compartment=None, mask_whole_cell=False,
    whole_nuclear_counts=None, whole_cyto_counts=None, whole_rng=None,
    whole_seed=None,
):
    """Return the V3 patch encoding plus one visible-only center whole cell.

    Raw center counts may be supplied by evaluation.  Training may omit them;
    the function then performs checked reconstruction from the center row of
    the float32 log1p cache.  The V3 branch remains pre-mask by design.
    """
    nuclear = np.asarray(nuclear_log)
    cyto = np.asarray(cyto_log)
    branch = encode_context_v3(
        nuclear, cyto, global_ids, node_indices=node_indices, seed=seed,
        epoch=epoch, slice_seed=slice_seed,
        context_tokens_per_compartment=context_tokens_per_compartment,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id,
        masked_genes=masked_genes, masked_compartment=masked_compartment,
        mask_whole_cell=mask_whole_cell,
    )
    if (whole_nuclear_counts is None) != (whole_cyto_counts is None):
        raise ValueError("Supply both raw whole-cell compartments or neither")
    if whole_nuclear_counts is None:
        if nuclear.ndim != 2 or cyto.ndim != 2 or not len(nuclear):
            raise ValueError("Cached context must contain a center row")
        whole_nuclear_counts = recover_raw_counts_v5(nuclear[0], name="nuclear center")
        whole_cyto_counts = recover_raw_counts_v5(cyto[0], name="cytoplasmic center")
    if whole_rng is None and whole_seed is None:
        nodes = (np.arange(len(nuclear), dtype=np.int64) if node_indices is None
                 else np.asarray(node_indices, dtype=np.int64))
        if nodes.shape != (len(nuclear),) or nodes[0] < 0:
            raise ValueError("Whole-cell encoding requires a real center node ID")
        whole_seed = np.random.SeedSequence(
            [int(seed), int(slice_seed), int(nodes[0]), 355])
    whole = encode_whole_cell_v5(
        whole_nuclear_counts, whole_cyto_counts, global_ids,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id,
        max_tokens=whole_cell_max_tokens, rng=whole_rng, seed=whole_seed,
        masked_genes=masked_genes, masked_compartment=masked_compartment,
        mask_whole_cell=mask_whole_cell,
    )
    branch.update(whole)
    return branch


class V5Dataset(V2Dataset):
    """V3's single-loss 70/20/10 task with a masked whole-cell center input."""

    def __init__(self, config: dict, split="train"):
        runtime_data = config["data"]
        requested_max_tokens = runtime_data.get("whole_cell_max_tokens", 1200)
        if (isinstance(requested_max_tokens, (bool, np.bool_))
                or int(requested_max_tokens) != requested_max_tokens
                or int(requested_max_tokens) < 1):
            raise ValueError("whole_cell_max_tokens must be a positive integer")
        self.whole_cell_max_tokens = int(requested_max_tokens)
        # This is a runtime packing budget, not a prepared-cache property.  V2
        # compares data_config exactly, so omit it from the inherited check.
        cache_config = copy.copy(config)
        cache_config["data"] = dict(runtime_data)
        cache_config["data"].pop("whole_cell_max_tokens", None)
        super().__init__(cache_config, split=split)
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

        output = encode_context_v5(
            nuclear, cyto, entry["genes"], node_indices=nodes, seed=seed,
            epoch=self.epoch, slice_seed=entry["slice_seed"],
            context_tokens_per_compartment=self.data["context_tokens_per_compartment"],
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            whole_cell_max_tokens=self.whole_cell_max_tokens,
            masked_genes=hidden_genes, masked_compartment=masked_compartment,
            mask_whole_cell=task_mode == 2,
        )
        full_bins = full_cell_bins_v3(
            nuclear[0], cyto[0], node_index=nodes[0], seed=seed,
            slice_seed=entry["slice_seed"])
        qids = np.full(2 * query_count, self.pad_id, dtype=np.int64)
        qcomp = np.repeat(np.arange(2, dtype=np.int64), query_count)
        qvalid = np.zeros(2 * query_count, dtype=bool)
        targets = np.full(2 * query_count, -2, dtype=np.int64)
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
            for compartment in (0, 1)
        ])
        output.update(
            coords=coords, query_gene_ids=qids, query_compartment_ids=qcomp,
            query_valid_mask=qvalid, target_bins=targets,
            target_nonzero=targets > 0, compartment_visible_mask=visible,
            task_mode=np.asarray(task_mode, dtype=np.int64),
            slice_index=np.asarray(slice_index, dtype=np.int64),
        )
        return output


def collate_whole_cell_v5(items, *, pad_token_id, pad_value=WHOLE_PAD_VALUE):
    """Pad variable-length whole-cell item fields to tensors ``[B, L]``."""
    if not items:
        raise ValueError("Cannot collate an empty whole-cell batch")
    lengths = []
    for item in items:
        arrays = [np.asarray(item[key]) for key in WHOLE_KEYS]
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Whole-cell item fields must be one-dimensional")
        if len({len(array) for array in arrays}) != 1 or not len(arrays[0]):
            raise ValueError("Whole-cell item fields must have matching nonempty lengths")
        if not arrays[2].all():
            raise ValueError("Item-level whole-cell sequences must be unpadded")
        lengths.append(len(arrays[0]))
    length = max(lengths)
    genes = np.full((len(items), length), int(pad_token_id), dtype=np.int64)
    bins = np.full((len(items), length), float(pad_value), dtype=np.float32)
    valid = np.zeros((len(items), length), dtype=bool)
    for row, (item, item_length) in enumerate(zip(items, lengths)):
        genes[row, :item_length] = item["whole_gene_ids"]
        bins[row, :item_length] = item["whole_input_bins"]
        valid[row, :item_length] = item["whole_token_valid_mask"]
    return {
        "whole_gene_ids": torch.from_numpy(genes),
        "whole_input_bins": torch.from_numpy(bins),
        "whole_token_valid_mask": torch.from_numpy(valid),
    }


def _infer_pad_token_id(batch):
    candidates = []
    for item in batch:
        invalid = ~np.asarray(item["token_valid_mask"], dtype=bool)
        if invalid.any():
            candidates.extend(np.asarray(item["gene_ids"])[invalid].tolist())
        if "target_bins" in item:
            query_pad = np.asarray(item["target_bins"]) == -2
            candidates.extend(np.asarray(item["query_gene_ids"])[query_pad].tolist())
    unique = np.unique(np.asarray(candidates, dtype=np.int64))
    if len(unique) != 1:
        raise ValueError("Cannot infer one pad token ID; pass pad_token_id to collate_v5")
    return int(unique[0])


def collate_v5(batch, *, pad_token_id=None):
    """Collate V3 patch/query fields and separately pad center whole cells."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    pad_token_id = (_infer_pad_token_id(batch) if pad_token_id is None
                    else int(pad_token_id))
    branch_items = [
        {key: value for key, value in item.items() if key not in WHOLE_KEYS}
        for item in batch
    ]
    output = collate_v2(branch_items)
    output.update(collate_whole_cell_v5(batch, pad_token_id=pad_token_id))
    return output
