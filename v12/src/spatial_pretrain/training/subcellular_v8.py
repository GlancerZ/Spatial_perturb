"""V8 objective: V4 masked 51-bin CE, optionally up-weighting nonzero targets.

With ``loss.nonzero_weight`` absent or 1 this is exactly the V4/V7 objective.
Otherwise each valid query gets weight ``w`` (target bin > 0) or 1 (zero), the
compartment mean becomes a weighted mean, and the cell/batch reduction is
unchanged. The unweighted CE is always reported as ``masked_ce``.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .subcellular_v4 import masked_bin_cross_entropy

LOSS_TYPE = 'masked_bin_cross_entropy'


def nonzero_weight(cfg) -> float:
    specification = cfg.get('loss', {})
    if not isinstance(specification, dict) or set(specification) - {'type', 'nonzero_weight'}:
        raise ValueError('V8 loss accepts only {type, nonzero_weight}')
    if specification.get('type') != LOSS_TYPE:
        raise ValueError(f'V8 requires loss.type={LOSS_TYPE}')
    weight = specification.get('nonzero_weight', 1.)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not weight >= 1:
        raise ValueError('loss.nonzero_weight must be a number >= 1')
    return float(weight)


def weighted_masked_bin_cross_entropy(logits, batch, weight):
    valid = batch['query_valid_mask'].bool()
    target = batch['target_bins']
    compartments = batch['query_compartment_ids']
    if not bool(valid.any()):
        return logits[valid].float().sum()
    per_query = logits.new_zeros(valid.shape, dtype=torch.float32).masked_scatter(
        valid, F.cross_entropy(logits[valid].float(), target[valid].long(), reduction='none'))
    weights = torch.where(target.gt(0), weight, 1.).float() * valid
    losses, present = [], []
    for compartment in (0, 1):
        selected = weights * compartments.eq(compartment)
        losses.append((per_query * selected).sum(-1) / selected.sum(-1).clamp_min(1e-12))
        present.append((valid & compartments.eq(compartment)).any(-1))
    losses, present = torch.stack(losses, -1), torch.stack(present, -1)
    per_cell = (losses * present).sum(-1) / present.sum(-1).clamp_min(1)
    return per_cell[present.any(-1)].mean()


def losses(output, batch, cfg, step):
    del step
    weight = nonzero_weight(cfg)
    plain = masked_bin_cross_entropy(output['joint_logits'], batch)
    total = plain if weight == 1. else weighted_masked_bin_cross_entropy(
        output['joint_logits'], batch, weight)
    return total, dict(total=float(total.detach()), masked_ce=float(plain.detach()))
