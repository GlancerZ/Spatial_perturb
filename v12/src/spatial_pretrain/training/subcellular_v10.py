"""Three alternatives for one fixed-teacher two-view latent prediction objective.

All targets are detached. Contrastive identities are (slice, node) pairs, so
repeated centers are positives rather than false negatives. Relational targets
soften false negatives with the teacher's similarities. Batches with only one
unique cell fall back explicitly to cosine prediction: a one-column softmax
would otherwise provide exactly zero learning signal.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _cosine(left: Tensor, right: Tensor) -> Tensor:
    return (1. - (F.normalize(left.float(), dim=-1, eps=1e-8)
                  * F.normalize(right.float(), dim=-1, eps=1e-8)).sum(-1)).mean()


def _same_identity(left: Tensor, right: Tensor) -> Tensor:
    return left[:, None, :].eq(right[None, :, :]).all(-1)


def _unique_first(identity: Tensor) -> Tensor:
    same = _same_identity(identity, identity)
    return ~same.tril(diagonal=-1).any(-1)


def _nce(query: Tensor, candidates: Tensor, query_ids: Tensor, candidate_ids: Tensor,
         temperature: float):
    positives = _same_identity(query_ids, candidate_ids)
    if not bool(positives.any(-1).all()):
        raise ValueError("Every contrastive anchor requires at least one same-cell positive")
    if int(_unique_first(candidate_ids).sum()) < 2:
        target = positives.float() @ candidates / positives.sum(-1, keepdim=True)
        return _cosine(query, target), 1
    logits = query.float() @ candidates.float().T / temperature
    positive_logits = logits.masked_fill(~positives, -torch.inf)
    # Sum the probability of every same-cell positive, not an arbitrary diagonal.
    return (logits.logsumexp(-1) - positive_logits.logsumexp(-1)).mean(), 0


def _relational(query: Tensor, row_targets: Tensor, bank: Tensor,
                student_temperature: float, teacher_temperature: float):
    if len(bank) < 2:
        return _cosine(query, row_targets), 1
    target_logp = (row_targets.float() @ bank.float().T / teacher_temperature).log_softmax(-1).detach()
    predicted_logp = (query.float() @ bank.float().T / student_temperature).log_softmax(-1)
    return F.kl_div(predicted_logp, target_logp, log_target=True, reduction="batchmean"), 0


def losses(output, batch, cfg, step=0):
    del step
    spec = cfg.get("loss", {})
    kind = spec.get("type", "cosine")
    if kind not in {"cosine", "contrastive", "relational"}:
        raise ValueError("V10 loss.type must be cosine, contrastive or relational")
    unknown = set(spec) - {"type", "temperature", "teacher_temperature", "student_temperature"}
    if unknown:
        raise ValueError(f"Unknown V10 loss options: {sorted(unknown)}")
    temperature = float(spec.get("temperature", .1))
    student_temperature = float(spec.get("student_temperature", temperature))
    teacher_temperature = float(spec.get("teacher_temperature", temperature))
    if any(not math.isfinite(value) or value <= 0 for value in
           (temperature, student_temperature, teacher_temperature)):
        raise ValueError("V10 temperatures must be finite and positive")
    nucleus, context = output["nucleus_prediction"].float(), output["context_prediction"].float()
    target = output["teacher"].detach().float()
    available = output["context_available"].bool()
    nucleus_available = output.get("nucleus_available", torch.ones_like(available)).bool()
    identity = batch["cell_identity"]
    if (nucleus.ndim != 2 or nucleus.shape != context.shape or nucleus.shape != target.shape
            or available.shape != (len(nucleus),) or nucleus_available.shape != (len(nucleus),)
            or identity.shape != (len(nucleus), 2)
            or not len(nucleus)):
        raise ValueError("V10 losses need aligned [B,D] views and [B,2] (slice,node) identities")
    keep = _unique_first(identity)
    paired = available & nucleus_available
    terms, fallback = {}, 0
    if kind == "cosine":
        if bool(nucleus_available.any()):
            terms["nucleus_teacher"] = _cosine(nucleus[nucleus_available], target[nucleus_available])
        if bool(available.any()):
            terms["context_teacher"] = _cosine(context[available], target[available])
        if bool(paired.any()):
            terms["view_consistency"] = _cosine(nucleus[paired], context[paired])
    elif kind == "contrastive":
        if bool(nucleus_available.any()):
            terms["nucleus_teacher"], count = _nce(
                nucleus[nucleus_available], target, identity[nucleus_available], identity, temperature)
            fallback += count
        if bool(available.any()):
            terms["context_teacher"], count = _nce(context[available], target, identity[available], identity,
                                                   temperature)
            fallback += count
        if bool(paired.any()):
            # Symmetric within the paired subset; both predictors receive gradients.
            forward, count_f = _nce(nucleus[paired], context[paired], identity[paired],
                                    identity[paired], temperature)
            backward, count_b = _nce(context[paired], nucleus[paired], identity[paired],
                                     identity[paired], temperature)
            terms["view_consistency"] = .5 * (forward + backward)
            fallback += count_f + count_b
    else:
        # Repeated centers must not alter the target class probabilities.
        bank = target[keep]
        if bool(nucleus_available.any()):
            terms["nucleus_teacher"], count = _relational(
                nucleus[nucleus_available], target[nucleus_available], bank,
                student_temperature, teacher_temperature)
            fallback += count
        if bool(available.any()):
            terms["context_teacher"], count = _relational(context[available], target[available], bank,
                                                          student_temperature, teacher_temperature)
            fallback += count
    # Both branches remain differentiable when absent. An all-empty batch has
    # exactly zero objective and no learning signal, rather than NaN or a
    # fabricated target for identical CLS-only nuclear inputs.
    zero = (nucleus.sum() + context.sum()) * 0.
    total = torch.stack(list(terms.values())).mean() + zero if terms else zero
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update(total=float(total.detach()), unique_cells=int(keep.sum()),
                   nucleus_cells=int(nucleus_available.sum()), context_cells=int(available.sum()),
                   paired_cells=int(paired.sum()), valid_loss_terms=len(terms),
                   has_learning_signal=bool(terms), degenerate_cosine_fallbacks=fallback)
    if bool(nucleus_available.any()):
        metrics["nucleus_teacher_cosine"] = float(
            (nucleus[nucleus_available] * target[nucleus_available]).sum(-1).mean().detach())
    if bool(available.any()):
        metrics["context_teacher_cosine"] = float((context[available] * target[available]).sum(-1).mean().detach())
    if bool(paired.any()):
        metrics["nucleus_context_cosine"] = float((nucleus[paired] * context[paired]).sum(-1).mean().detach())
    return total, metrics
