"""Single NB-total × beta-binomial-allocation likelihood for V12 views.

Genes are averaged per cell, cells per mode, then available modes are averaged.
Whole-update global cell counts give exactly equivalent accumulation and DDP
normalization. N/C are paired raw integer counts; bins are encoding inputs only.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from ..models.subcellular_v12 import MODES
from .subcellular_v11 import reduce_loss_sums


def _loss_options(cfg):
    spec = cfg.get("loss", {})
    if set(spec) - {"type", "modes"}:
        raise ValueError(f"Unknown V12 loss options: {sorted(set(spec) - {'type', 'modes'})}")
    if spec.get("type", "joint_compartment_count_nll") not in {
            "joint_compartment_count_nll", "joint_count_nll", "nb_beta_binomial"}:
        raise ValueError("V12 uses one joint total/allocation count likelihood")
    modes = tuple(spec.get("modes", MODES))
    if not modes or len(set(modes)) != len(modes) or any(mode not in MODES for mode in modes):
        raise ValueError("loss.modes must be a nonempty unique subset of the five V12 modes")
    return modes


def validate_targets(batch):
    valid = batch["query_valid_mask"].bool()
    if valid.ndim != 2:
        raise ValueError("Query validity requires [B,Q]")
    values = [batch[f"query_{name}_counts"] for name in ("nucleus", "cytoplasm", "total")]
    for name, value in zip(("nucleus", "cytoplasm", "total"), values):
        if value.shape != valid.shape:
            raise ValueError(f"{name} counts must align with [B,Q] query mask")
        if not bool(torch.isfinite(value).all()) or bool(value.lt(0).any()):
            raise ValueError("Count targets including padding must be finite and nonnegative")
        if bool(value.double().ne(value.double().round()).any()):
            raise ValueError("V12 targets must be raw integer counts")
        if bool(value[~valid].ne(0).any()):
            raise ValueError("Padding target counts must be zero, not hidden real measurements")
        if bool(value.gt(2**53 - 1).any()):
            raise ValueError("Counts exceed exact float64 integer support")
    nucleus, cytoplasm, total = [value.double() for value in values]
    if not bool((nucleus + cytoplasm).eq(total).all()):
        raise ValueError("Paired nuclear and cytoplasmic counts must satisfy N + C = total exactly")
    return nucleus, cytoplasm, total, valid


def joint_count_log_prob(nucleus: Tensor, cytoplasm: Tensor, parameters: Mapping[str, Tensor]):
    """Log p(T=N+C) + log p(N|T), computed in float64 for lgamma accuracy.

    This helper expects validated support. The public loss validates targets and
    decoder parameters before calling it; double arithmetic retains gradients.
    """
    n, c = nucleus.double(), cytoplasm.double()
    total = n + c
    mu, theta, p, kappa = [parameters[name].double() for name in (
        "mu_total", "theta_total", "p_nucleus", "kappa_allocation")]
    log_denominator = (theta + mu).log()
    nb = (torch.lgamma(total + theta) - torch.lgamma(theta) - torch.lgamma(total + 1)
          + theta * (theta.log() - log_denominator) + total * (mu.log() - log_denominator))
    alpha, beta = p * kappa, (1 - p) * kappa
    allocation = (torch.lgamma(total + 1) - torch.lgamma(n + 1) - torch.lgamma(c + 1)
                  + torch.lgamma(n + alpha) + torch.lgamma(c + beta) - torch.lgamma(total + kappa)
                  + torch.lgamma(kappa) - torch.lgamma(alpha) - torch.lgamma(beta))
    # At T=0, allocation is exactly deterministic and provides no p/kappa signal.
    allocation = torch.where(total.eq(0), torch.zeros_like(allocation), allocation)
    return nb + allocation


def joint_count_nll(parameters: Mapping[str, Tensor], nuclear: Tensor, total: Tensor):
    """Elementwise joint NLL for evaluation; input tensors must have count support.

    Do not average padded genes or use validation targets as model inputs. The
    training entry point performs additional target/mask/parameter validation.
    """
    return -joint_count_log_prob(nuclear, total - nuclear, parameters)


def loss_sums_and_counts(output, batch, cfg):
    modes = _loss_options(cfg)
    nucleus, cytoplasm, total, valid = validate_targets(batch)
    parameters = output["count_parameters"]
    required = {"mu_total", "theta_total", "p_nucleus", "kappa_allocation"}
    zero = sum(value.sum() * 0. for item in parameters.values() for value in item.values())
    genes_per_cell = valid.sum(-1)
    sums, counts = {}, {}
    for mode in modes:
        item = parameters[mode]
        if set(item) != required:
            raise ValueError("Each mode must predict exactly the four joint count parameters")
        available = output["mode_available"][mode].bool()
        if available.shape != (len(valid),):
            raise ValueError("Mode availability requires [B]")
        for name, value in item.items():
            if value.shape != valid.shape or not bool(torch.isfinite(value).all()) or bool(value.le(0).any()):
                raise ValueError(f"{name} requires finite positive [B,Q] parameters")
        if bool(item["p_nucleus"].ge(1).any()):
            raise ValueError("p_nucleus must lie strictly between zero and one")
        error = -joint_count_log_prob(nucleus, cytoplasm, item)
        if not bool(torch.isfinite(error).all()):
            raise ValueError("Joint count likelihood produced a nonfinite value")
        cell_loss = torch.where(valid, error, 0.).sum(-1) / genes_per_cell.clamp_min(1)
        eligible = available & genes_per_cell.gt(0)
        sums[mode] = (cell_loss * eligible).sum() + zero
        counts[mode] = eligible.sum().detach()
    return sums, counts


def loss_counts_from_batch(batch, cfg, *, mode_available: Mapping[str, Tensor]):
    modes = _loss_options(cfg)
    valid = batch["query_valid_mask"].bool()
    if valid.ndim != 2:
        raise ValueError("Query validity requires [B,Q]")
    cells = valid.any(-1)
    result = {}
    for mode in modes:
        mask = mode_available[mode].bool()
        if mask.shape != cells.shape:
            raise ValueError("Mode availability must align to query cells")
        result[mode] = (cells & mask).sum().detach()
    return result


def losses(output, batch, cfg, step=0, *, global_counts=None, world_size=1):
    del step
    sums, counts = loss_sums_and_counts(output, batch, cfg)
    total = reduce_loss_sums(sums, counts, global_counts=global_counts, world_size=world_size)
    metrics = {f"{mode}_nll": float((sums[mode] / counts[mode].clamp_min(1)).detach()) for mode in sums}
    metrics.update({f"{mode}_cells": int(counts[mode]) for mode in sums})
    denominators = counts if global_counts is None else global_counts
    metrics.update(total=float(total.detach()), valid_loss_terms=sum(int(x) > 0 for x in denominators.values()),
                   has_learning_signal=any(int(x) > 0 for x in denominators.values()),
                   valid_query_genes=int(batch["query_valid_mask"].sum()))
    return total, metrics
