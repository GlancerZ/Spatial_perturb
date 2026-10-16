"""The single V4 objective: balanced masked 51-bin categorical cross entropy."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_bin_cross_entropy(logits, batch):
    """Mean queries per valid compartment, then valid compartments, then cells.

    Bin zero is a measured zero and receives ordinary CE. PAD queries are
    excluded before target conversion; negative, fractional, non-finite or
    out-of-range VALID targets are rejected rather than silently clamped.
    """
    valid = batch['query_valid_mask'].bool()
    target = batch['target_bins']
    compartments = batch['query_compartment_ids']
    if (valid.ndim != 2 or target.shape != valid.shape or compartments.shape != valid.shape
            or logits.shape != (*valid.shape, 51)):
        raise ValueError('Expected target/query/compartment [B,Q] and logits [B,Q,51]')
    bad_target = (~torch.isfinite(target) | target.lt(0) | target.gt(50)
                  | target.ne(target.round()))
    if bool((bad_target & valid).any()):
        raise ValueError('Valid V4 target bins must be integer ranks in [0,50]')
    bad_compartment = compartments.ne(0) & compartments.ne(1)
    if bool((bad_compartment & valid).any()):
        raise ValueError('Valid query compartments must be nuclear=0 or cytoplasm=1')
    # Selecting active queries also keeps invalid logits (possibly NaN in an
    # upstream padding buffer) outside CE, and gives PAD exactly zero gradient.
    if not bool(valid.any()):
        return logits[valid].float().sum()
    active_loss = F.cross_entropy(logits[valid].float(), target[valid].long(), reduction='none')
    per_query = active_loss.new_zeros(valid.shape).masked_scatter(valid, active_loss)
    losses, present = [], []
    for compartment in (0, 1):
        selected = valid & compartments.eq(compartment)
        count = selected.sum(-1)
        losses.append((per_query * selected).sum(-1) / count.clamp_min(1))
        present.append(count.gt(0))
    losses, present = torch.stack(losses, -1), torch.stack(present, -1)
    per_cell = (losses * present).sum(-1) / present.sum(-1).clamp_min(1)
    active_cell = present.any(-1)
    return per_cell[active_cell].mean()


def losses(output, batch, cfg, step):
    """No auxiliary heads, BCE, rank MSE, or configurable loss mixtures."""
    del step
    specification = cfg.get('loss', {})
    if not isinstance(specification, dict) or set(specification) - {'type'}:
        raise ValueError('V4 supports only loss: {type: masked_bin_cross_entropy}')
    if specification.get('type', 'masked_bin_cross_entropy') != 'masked_bin_cross_entropy':
        raise ValueError('V4 requires the single masked_bin_cross_entropy objective')
    total = masked_bin_cross_entropy(output['joint_logits'], batch)
    value = float(total.detach())
    return total, dict(total=value, masked_ce=value)
