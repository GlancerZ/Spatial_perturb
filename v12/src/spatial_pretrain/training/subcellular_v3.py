"""CLS-only nonzero and normalized positive-bin reconstruction for V3."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def binned_reconstruction_loss(prediction, batch, mask=None):
    """Average queries within compartments, compartments within cells, then cells.

    Real zero bins supervise BCE only. Positive bins 1..50 supervise rank in
    [0,1]. Invalid queries never enter either denominator or target arithmetic.
    """
    valid = batch['query_valid_mask'].bool()
    if mask is not None:
        valid = valid & mask.bool()
    target = batch['target_bins'].float()
    if target.shape != valid.shape or prediction.shape != (*valid.shape, 2):
        raise ValueError('Expected target/query mask [B,Q] and predictions [B,Q,2]')
    bad = ~torch.isfinite(target) | (target < 0) | (target > 50) | (target != target.round())
    if bool((bad & valid).any()):
        raise ValueError('Valid V3 target bins must be integer ranks in [0,50]')
    target = torch.where(valid, target, torch.zeros_like(target)).float()
    nonzero = target > 0
    bce = F.binary_cross_entropy_with_logits(prediction[..., 0].float(), nonzero.float(), reduction='none')
    rank = (target - 1.) / 49.
    error = (prediction[..., 1].float().sigmoid() - rank).square()
    values, present = [], []
    for compartment in (0, 1):
        selected = valid & batch['query_compartment_ids'].eq(compartment)
        positive = selected & nonzero
        count = selected.sum(-1)
        loss = (bce * selected).sum(-1) / count.clamp_min(1)
        loss = loss + (error * positive).sum(-1) / positive.sum(-1).clamp_min(1)
        values.append(loss)
        present.append(count > 0)
    values, present = torch.stack(values, -1), torch.stack(present, -1)
    per_cell = (values * present).sum(-1) / present.sum(-1).clamp_min(1)
    active = present.any(-1)
    return (per_cell * active).sum() / active.sum().clamp_min(1)


def losses(output, batch, cfg, step):
    """Original visible-compartment CLS loss plus final joint CLS loss."""
    del step  # Both losses use a fixed target definition and fixed initial weights.
    original_valid = batch['token_valid_mask'][:, 0].bool()
    original_compartment = batch['compartment_ids'][:, 0]
    visible = torch.stack([(original_valid & original_compartment.eq(index)).any(-1)
                           for index in (0, 1)], -1)
    comp_mask = visible.gather(1, batch['query_compartment_ids'])
    compartment = binned_reconstruction_loss(output['compartment_pred'], batch, comp_mask)
    joint = binned_reconstruction_loss(output['joint_pred'], batch)
    weights = cfg.get('loss', {})
    total = weights.get('compartment', 1.) * compartment + weights.get('joint', 1.) * joint
    return total, dict(total=float(total.detach()), compartment=float(compartment.detach()),
                       joint=float(joint.detach()))
