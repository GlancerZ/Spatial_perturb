"""One expression-bin task under nuclear, spatial and fused input conditions.

Reduction order is valid genes -> cells within each mode -> available modes.
DDP callers must SUM local counts across ranks (and accumulation microbatches)
then pass those ``global_counts`` and ``world_size`` to the reduction. DDP
averages gradients, so local sums are multiplied by world_size. Do NOT divide
again by accumulation_steps when counts already span the accumulation window.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

MODES = ("nucleus", "spatial", "fused")


def _loss_options(cfg):
    spec = cfg.get("loss", {})
    unknown = set(spec) - {"type", "modes", "label_smoothing"}
    if unknown:
        raise ValueError(f"Unknown V11 loss options: {sorted(unknown)}")
    if spec.get("type", "bin_ce") not in {"bin_ce", "expression_ce", "masked_bin_cross_entropy"}:
        raise ValueError("V11 has one expression-bin CE task, no teacher/alignment objective")
    modes = tuple(spec.get("modes", MODES))
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("loss.modes must be a nonempty unique subset of nucleus/spatial/fused")
    smoothing = float(spec.get("label_smoothing", 0.))
    if not 0 <= smoothing < 1:
        raise ValueError("label_smoothing must lie in [0,1)")
    return modes, smoothing


def loss_sums_and_counts(output, batch, cfg):
    """Return ({mode: differentiable cell-loss sum}, {mode: detached cell count})."""
    modes, smoothing = _loss_options(cfg)
    targets, valid = batch["query_bins"], batch["query_valid_mask"].bool()
    if targets.ndim != 2 or valid.shape != targets.shape:
        raise ValueError("Expression targets and valid gene mask must share [B,Q]")
    if not bool(torch.isfinite(targets[valid]).all()):
        raise ValueError("Valid target bins must be finite")
    if bool((targets[valid] != targets[valid].long()).any()):
        raise ValueError("Expression targets must be integer bin IDs, not absolute counts")
    safe_targets = torch.where(valid, targets, 0).long()
    genes_per_cell = valid.sum(-1)
    sums, counts = {}, {}
    # Include every prediction in a zero graph so mode ablations need not enable
    # DDP find_unused_parameters solely because another decoder output is omitted.
    all_logits = output["logits"]
    zero = sum(value.sum() * 0. for value in all_logits.values())
    for mode in modes:
        logits = all_logits[mode]
        available = output["mode_available"][mode].bool()
        if logits.ndim != 3 or logits.shape[:2] != targets.shape or available.shape != (len(targets),):
            raise ValueError("V11 requires logits [B,Q,bins] and mode availability [B]")
        if bool(((safe_targets < 0) | (safe_targets >= logits.shape[-1])).any()):
            raise ValueError("Target bin out of decoder range")
        # Q=0 supports empty panels without passing an empty matrix to CE.
        if targets.shape[1]:
            errors = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                     safe_targets.reshape(-1), reduction="none",
                                     label_smoothing=smoothing).reshape_as(targets)
            cell_loss = torch.where(valid, errors, 0.).sum(-1) / genes_per_cell.clamp_min(1)
        else:
            cell_loss = logits.sum(dim=(-1, -2)) * 0.
        eligible = available & genes_per_cell.gt(0)
        sums[mode] = (cell_loss * eligible).sum() + zero
        counts[mode] = eligible.sum().detach()
    return sums, counts


def loss_counts_from_batch(batch, cfg, *, mode_available: Mapping[str, Tensor]):
    """Count cells before forward using ``model.mode_availability(full_batch)``.

    Pass the complete local optimizer-update batch, then all-reduce the counts.
    Keeping availability in the model guarantees graph radius, cell identities,
    token padding and missing branches match the actual forward path.
    """
    modes, _ = _loss_options(cfg)
    query_valid = batch["query_valid_mask"].bool()
    if query_valid.ndim != 2:
        raise ValueError("Query validity must be [B,Q]")
    cells = query_valid.any(-1)
    result = {}
    for mode in modes:
        mask = mode_available[mode].bool()
        if mask.shape != cells.shape:
            raise ValueError("Mode availability must align to query cells")
        result[mode] = (cells & mask).sum().detach()
    return result


def reduce_loss_sums(sums: Mapping[str, Tensor], counts: Mapping[str, Tensor], *,
                     global_counts: Mapping[str, Tensor] | None = None, world_size: int = 1):
    """Reduce local sums using either local or explicitly all-reduced counts."""
    if not sums or set(sums) != set(counts):
        raise ValueError("Sums and counts must have the same nonempty mode keys")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    denominator = counts if global_counts is None else global_counts
    if set(denominator) != set(sums):
        raise ValueError("global_counts must include exactly the selected loss modes")
    if global_counts is None and world_size != 1:
        raise ValueError("Distributed reduction requires global_counts")
    for value in denominator.values():
        if not bool(torch.isfinite(torch.as_tensor(value)).all()) or bool(torch.as_tensor(value).lt(0).any()):
            raise ValueError("Counts must be finite nonnegative scalars")
    active = [mode for mode in sums if int(denominator[mode]) > 0]
    zero = sum(sums.values()) * 0.
    if not active:
        return zero
    return sum(sums[mode] * world_size / denominator[mode] for mode in active) / len(active) + zero


def losses(output, batch, cfg, step=0, *, global_counts=None, world_size=1):
    del step
    sums, counts = loss_sums_and_counts(output, batch, cfg)
    total = reduce_loss_sums(sums, counts, global_counts=global_counts, world_size=world_size)
    metrics = {f"{mode}_ce": float((sums[mode] / counts[mode].clamp_min(1)).detach()) for mode in sums}
    metrics.update({f"{mode}_cells": int(counts[mode]) for mode in sums})
    denominators = counts if global_counts is None else global_counts
    metrics.update(total=float(total.detach()), valid_loss_terms=sum(int(x) > 0 for x in denominators.values()),
                   has_learning_signal=any(int(x) > 0 for x in denominators.values()),
                   valid_query_genes=int(batch["query_valid_mask"].sum()))
    return total, metrics
