"""V6: one visible-data view for compartment, whole-cell, and LR inputs.

All copies of the physical center receive the same mask before input binning.
The unmasked V3 full-cell bin table is used only for supervision; input ranks
and target ranks therefore need not agree. The ordinary whole-cell branch
retains V5's packing, tie randomization, and paired-baseline seed contract.

LR is an additional *visible* side channel: it may read the entire measured
cytoplasmic panel, including genes omitted by the encoder token budget. Panel
membership comes from global vocabulary IDs, never from positive RNA tokens.
Observed zero remains known; a masked or unmeasured endpoint is unknown.
No hidden count contributes to LR activity, its depth, or its coverage.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

from .subcellular_v2 import vocabulary_path
from .subcellular_v3 import (
    encode_context_v3,
    full_cell_bins_v3,
    sample_paired_queries_v3,
    sample_uniform_queries_v3,
)
from .subcellular_v5 import (
    V5Dataset,
    _raw_count_vector,
    collate_v5,
    encode_whole_cell_v5,
    recover_raw_counts_v5,
)


LR_KEYS = ("lr_bias", "lr_coverage")
LR_DEPTH_SCALE = 10_000.0
LR_RELIABILITY_PSEUDO_WEIGHT = 4.0
_GENE_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9.-]*\Z")


def load_lr_prior(config, vocab=None):
    """Load a frozen single-gene CSV; return ``pairs, weights, metadata`` dict.

    ``config`` is the complete experiment config with top-level ``lr`` keys
    ``enabled``, ``path``, ``sha256``, and ``source``. ``vocab`` may be an
    already loaded gene-to-ID mapping; otherwise use ``assets_dir``. CSV must
    contain ``ligand,receptor`` and optionally confidence ``weight`` in (0,1].
    Compound syntax and duplicate directed pairs are rejected. Out-of-vocab
    pairs are explicitly counted in metadata and never interpreted as zero.
    The SHA-256 is checked before parsing. LR metadata is runtime information
    and must not be inserted into V2's exactly compared ``data_config``.
    """
    prior = config.get("lr", {})
    if not isinstance(prior, dict):
        raise TypeError("config['lr'] must be a mapping")
    enabled = prior.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("lr.enabled must be boolean")
    empty = {"pairs": np.empty((0, 2), dtype=np.int64),
             "weights": np.empty(0, dtype=np.float32)}
    if not enabled:
        return dict(empty, metadata={"enabled": False, "loaded_pairs": 0})
    for key in ("path", "sha256", "source"):
        if not isinstance(prior.get(key), str) or not prior[key].strip():
            raise ValueError(f"Enabled LR requires a nonempty lr.{key}")
    expected = prior["sha256"].lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("lr.sha256 must contain exactly 64 hexadecimal characters")
    path = Path(prior["path"])
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"LR prior SHA-256 mismatch for {path}: {actual} != {expected}")
    if vocab is None:
        vocab = json.loads(vocabulary_path(config["assets_dir"]).read_text())
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if not reader.fieldnames or not {"ligand", "receptor"}.issubset(reader.fieldnames):
        raise ValueError("LR CSV requires ligand and receptor columns")
    pairs, weights, unmapped, seen = [], [], [], set()
    rows = 0
    for line, row in enumerate(reader, start=2):
        rows += 1
        names = tuple((row.get(key) or "").strip() for key in ("ligand", "receptor"))
        if not all(_GENE_SYMBOL.fullmatch(name) for name in names):
            raise ValueError(f"LR CSV line {line}: only single-gene symbols are supported; no compounds")
        if names in seen:
            raise ValueError(f"LR CSV line {line}: duplicate directed pair {names}")
        seen.add(names)
        try:
            weight = float(row.get("weight", "1"))
        except (ValueError, TypeError) as error:
            raise ValueError(f"LR CSV line {line}: invalid confidence weight") from error
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError(f"LR CSV line {line}: confidence weight must lie in (0, 1]")
        if not all(name in vocab for name in names):
            unmapped.append(list(names))
            continue
        ids = tuple(int(vocab[name]) for name in names)
        if min(ids) < 0:
            raise ValueError(f"LR CSV line {line}: vocabulary IDs must be nonnegative")
        pairs.append(ids)
        weights.append(weight)
    if not rows or not pairs:
        raise ValueError("Enabled LR prior contains no single-gene pairs mapped to the vocabulary")
    mapped = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(np.unique(mapped, axis=0)) != len(mapped):
        raise ValueError("LR symbols alias duplicate directed vocabulary-ID pairs")
    return {
        "pairs": mapped,
        "weights": np.asarray(weights, dtype=np.float32),
        "metadata": {
            "enabled": True, "path": str(path.resolve()), "sha256": actual,
            "source": prior["source"], "csv_rows": rows,
            "loaded_pairs": len(pairs), "unmapped_pairs": unmapped,
            "unmapped_pair_count": len(unmapped), "pair_semantics": "single_gene_only",
            "orientation": "receiver_i_sender_j",
            "expression": "visible_cytoplasmic_raw_counts_full_measured_panel",
            "depth_scale": LR_DEPTH_SCALE,
            "bias": "weighted_activity_sum_divided_by_known_weight_plus_4",
            "reliability_pseudo_weight": LR_RELIABILITY_PSEUDO_WEIGHT,
            "coverage": "known_weight_divided_by_full_prior_weight_diagnostic_only",
        },
    }


def _visible_view(nuclear_log, cyto_log, global_ids, node_indices,
                  masked_genes, masked_compartment, mask_whole_cell):
    nuclear = np.array(nuclear_log, dtype=np.float32, copy=True)
    cyto = np.array(cyto_log, dtype=np.float32, copy=True)
    genes = np.asarray(global_ids, dtype=np.int64)
    if nuclear.ndim != 2 or nuclear.shape != cyto.shape or not all(nuclear.shape):
        raise ValueError("Expected matching nonempty [cells, panel_genes] compartment arrays")
    cells, panel_size = nuclear.shape
    if genes.shape != (panel_size,) or len(np.unique(genes)) != panel_size:
        raise ValueError("Each measured panel column needs a unique vocabulary ID")
    nodes = (np.arange(cells, dtype=np.int64) if node_indices is None
             else np.asarray(node_indices, dtype=np.int64))
    if nodes.shape != (cells,) or nodes[0] < 0:
        raise ValueError("node_indices must contain one ID per row and a real center")
    if masked_compartment not in (None, 0, 1):
        raise ValueError("masked_compartment must be nuclear=0 or cytoplasmic=1")
    known = np.broadcast_to((nodes >= 0)[:, None, None], (cells, 2, panel_size)).copy()
    center_rows = np.flatnonzero(nodes == nodes[0])
    if masked_genes is not None:
        hidden = np.asarray(masked_genes, dtype=np.int64)
        if (hidden.ndim != 1 or (hidden < 0).any() or (hidden >= panel_size).any()
                or len(np.unique(hidden)) != len(hidden)):
            raise ValueError("masked_genes must be unique measured-panel column indices")
        known[np.ix_(center_rows, np.arange(2), hidden)] = False
    if masked_compartment is not None:
        known[center_rows, masked_compartment, :] = False
    if mask_whole_cell:
        known[center_rows] = False
    nuclear[~known[:, 0]] = 0
    cyto[~known[:, 1]] = 0
    if (not np.isfinite(nuclear).all() or not np.isfinite(cyto).all()
            or (nuclear < 0).any() or (cyto < 0).any()):
        raise ValueError("Visible cached log counts must be finite and nonnegative")
    return nuclear, cyto, genes, nodes, known


def _lr_evidence(visible_cyto_raw, known_cyto, genes, nodes, lr_pairs, lr_weights):
    """Return bounded receiver-by-sender evidence and confidence coverage.

    For each cell, scale visible counts by 1e4 / sum(visible counts), then use
    x/(x+1). For each directed edge, average ligand_sender * receptor_receiver
    over known pairs with confidence weights, and shrink by known_weight /
    (known_weight + 4). The fixed pseudoweight 4 prevents a lone observed pair
    from acting as high-confidence evidence, without dividing useful signal
    by the full database size. lr_coverage is independently known_weight /
    full_prior_weight, for diagnostics only. Observed-zero pairs contribute
    known weight; unknown pairs do not. PAD and self/duplicate-self edges are
    zero; the model applies physical-neighbor radius eligibility separately.
    """
    cells = len(nodes)
    bias = np.zeros((cells, cells), dtype=np.float32)
    coverage = np.zeros_like(bias)
    if lr_pairs is None:
        if lr_weights is not None:
            raise ValueError("lr_weights requires lr_pairs")
        return {"lr_bias": bias, "lr_coverage": coverage}
    pairs = np.asarray(lr_pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("lr_pairs must have shape [pairs, 2] in global vocabulary IDs")
    if not np.issubdtype(pairs.dtype, np.integer) or (pairs < 0).any():
        raise ValueError("lr_pairs must contain nonnegative integer vocabulary IDs")
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError("lr_pairs must contain unique directed pairs")
    weights = (np.ones(len(pairs), dtype=np.float64) if lr_weights is None
               else np.asarray(lr_weights, dtype=np.float64))
    if (weights.shape != (len(pairs),) or not np.isfinite(weights).all()
            or (weights <= 0).any() or (weights > 1).any()):
        raise ValueError("lr_weights must be matching finite confidences in (0, 1]")
    if not len(pairs):
        return {"lr_bias": bias, "lr_coverage": coverage}
    panel = {int(gene): column for column, gene in enumerate(genes)}
    measured = np.array([int(lig) in panel and int(rec) in panel for lig, rec in pairs])
    if not measured.any():
        return {"lr_bias": bias, "lr_coverage": coverage}
    selected = pairs[measured]
    ligand = np.array([panel[int(gene)] for gene in selected[:, 0]], dtype=np.int64)
    receptor = np.array([panel[int(gene)] for gene in selected[:, 1]], dtype=np.int64)
    raw = visible_cyto_raw.astype(np.float64)
    depth = raw.sum(axis=1, keepdims=True)
    scaled = np.divide(raw * LR_DEPTH_SCALE, depth, out=np.zeros_like(raw), where=depth > 0)
    activity = scaled / (scaled + 1.0)
    sender_known = known_cyto[:, ligand].astype(np.float64)
    receiver_known = known_cyto[:, receptor].astype(np.float64)
    selected_weights = weights[measured]
    total_weight = weights.sum()
    # Rows: receiver, columns: sender. Multiplication does not use RNA targets.
    known_weight = (receiver_known * selected_weights) @ sender_known.T
    coverage[:] = known_weight / total_weight
    bias[:] = (((receiver_known * activity[:, receptor]) * selected_weights)
               @ (sender_known * activity[:, ligand]).T) / (known_weight + LR_RELIABILITY_PSEUDO_WEIGHT)
    eligible = ((nodes[:, None] >= 0) & (nodes[None, :] >= 0)
                & (nodes[:, None] != nodes[None, :]))
    bias[~eligible] = 0
    coverage[~eligible] = 0
    np.clip(bias, 0, 1, out=bias)
    np.clip(coverage, 0, 1, out=coverage)
    return {"lr_bias": bias, "lr_coverage": coverage}


def compute_lr_features_v6(cyto_raw_counts, global_ids, *, node_indices,
                           visible_mask=None, lr_pairs=None, lr_weights=None):
    """Compute LR for one small patch, directly from discrete raw RNA counts.

    ``visible_mask`` is [nodes, panel_genes], with True for measured, unmasked
    entries including zeros. It defaults to all panel genes of real nodes.
    Evaluation should call this per spatial patch, not for all unique cells:
    output arrays are [nodes,nodes]. The caller must apply the center mask to
    *all* duplicate center rows in visible_mask, as encode_context_v6 does.
    """
    raw = np.asarray(cyto_raw_counts)
    genes = np.asarray(global_ids, dtype=np.int64)
    nodes = np.asarray(node_indices, dtype=np.int64)
    if raw.ndim != 2 or not all(raw.shape):
        raise ValueError("cyto_raw_counts must be a nonempty [nodes, panel_genes] array")
    if genes.shape != (raw.shape[1],) or len(np.unique(genes)) != len(genes):
        raise ValueError("Each measured panel column needs a unique vocabulary ID")
    if nodes.shape != (len(raw),):
        raise ValueError("node_indices must contain one ID per row")
    known = (np.ones(raw.shape, dtype=bool) if visible_mask is None
             else np.asarray(visible_mask, dtype=bool).copy())
    if known.shape != raw.shape:
        raise ValueError("visible_mask must match cyto_raw_counts")
    known[nodes < 0] = False
    visible_raw = np.where(known, raw, 0)
    visible_raw = _raw_count_vector(visible_raw.reshape(-1), "visible cytoplasmic LR counts").reshape(raw.shape)
    return _lr_evidence(visible_raw, known, genes, nodes, lr_pairs, lr_weights)


def encode_context_v6(
    nuclear_log, cyto_log, global_ids, *, node_indices=None, seed=42, epoch=0,
    slice_seed=0, context_tokens_per_compartment=256, pad_token_id,
    cls_token_id, whole_cell_max_tokens=1200, masked_genes=None,
    masked_compartment=None, mask_whole_cell=False,
    whole_nuclear_counts=None, whole_cyto_counts=None, whole_rng=None,
    whole_seed=None, lr_pairs=None, lr_weights=None,
):
    """V5-compatible encoder API with a shared center mask and directed LR.

    ``lr_pairs[:, 0]`` are ligands and ``lr_pairs[:, 1]`` are receptors, in
    global scGPT vocabulary IDs. Both new float32 fields have shape [N,N],
    indexed [receiver, sender]. ``lr_bias`` includes evidence reliability
    shrinkage. ``lr_coverage`` records full-prior coverage for diagnostics;
    the model must not multiply the bias by that diagnostic coverage.
    """
    nuclear, cyto, genes, nodes, known = _visible_view(
        nuclear_log, cyto_log, global_ids, node_indices,
        masked_genes, masked_compartment, mask_whole_cell)
    # V3's pre-binning now sees only visible RNA, including for every copy of
    # the center. Its independent per-node bins/token RNG streams are retained.
    output = encode_context_v3(
        nuclear, cyto, genes, node_indices=nodes, seed=seed, epoch=epoch,
        slice_seed=slice_seed,
        context_tokens_per_compartment=context_tokens_per_compartment,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id)
    if (whole_nuclear_counts is None) != (whole_cyto_counts is None):
        raise ValueError("Supply both raw whole-cell compartments or neither")
    if whole_nuclear_counts is None:
        whole_nuclear_counts = recover_raw_counts_v5(nuclear[0], name="visible nuclear center")
        whole_cyto_counts = recover_raw_counts_v5(cyto[0], name="visible cytoplasmic center")
    if whole_rng is None and whole_seed is None:
        whole_seed = np.random.SeedSequence([int(seed), int(slice_seed), int(nodes[0]), 355])
    output.update(encode_whole_cell_v5(
        whole_nuclear_counts, whole_cyto_counts, genes,
        pad_token_id=pad_token_id, cls_token_id=cls_token_id,
        max_tokens=whole_cell_max_tokens, rng=whole_rng, seed=whole_seed,
        masked_genes=masked_genes, masked_compartment=masked_compartment,
        mask_whole_cell=mask_whole_cell))
    visible_raw = (None if lr_pairs is None or np.size(lr_pairs) == 0 else
                   recover_raw_counts_v5(cyto, name="visible cytoplasmic LR counts"))
    output.update(_lr_evidence(visible_raw, known[:, 1], genes, nodes, lr_pairs, lr_weights))
    return output


class V6Dataset(V5Dataset):
    """Validated V2 cache, V5 paired tasks, visible inputs, full-bin targets."""

    def __init__(self, config: dict, split="train"):
        super().__init__(config, split=split)
        prior = load_lr_prior(config)
        self.lr_pairs = prior["pairs"]
        self.lr_weights = prior["weights"]
        self.lr_metadata = prior["metadata"]

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
        output = encode_context_v6(
            nuclear, cyto, entry["genes"], node_indices=nodes, seed=seed,
            epoch=self.epoch, slice_seed=entry["slice_seed"],
            context_tokens_per_compartment=self.data["context_tokens_per_compartment"],
            pad_token_id=self.pad_id, cls_token_id=self.cls_id,
            whole_cell_max_tokens=self.whole_cell_max_tokens,
            masked_genes=hidden_genes, masked_compartment=masked_compartment,
            mask_whole_cell=task_mode == 2,
            lr_pairs=self.lr_pairs, lr_weights=self.lr_weights)
        # Unmasked values are consulted only below, for supervised targets.
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
            for compartment in (0, 1)])
        output.update(
            coords=coords, query_gene_ids=qids, query_compartment_ids=qcomp,
            query_valid_mask=qvalid, target_bins=targets,
            target_nonzero=targets > 0, compartment_visible_mask=visible,
            task_mode=np.asarray(task_mode, dtype=np.int64),
            slice_index=np.asarray(slice_index, dtype=np.int64))
        return output


def collate_v6(batch, *, pad_token_id=None):
    """Reuse V5 packing while stacking LR arrays without token-axis trimming."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    for item in batch:
        cells = len(item["node_indices"])
        for key in LR_KEYS:
            value = np.asarray(item[key])
            if value.shape != (cells, cells) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite [nodes, nodes] array")
            if (value < 0).any() or (value > 1).any():
                raise ValueError(f"{key} must lie in [0, 1]")
    # collate_v2 (called inside collate_v5) trims only named token fields.
    return collate_v5(batch, pad_token_id=pad_token_id)
