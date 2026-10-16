"""V9 objective: two reconstruction CEs plus a small view-consistency term.

total = CE(self view) + CE(context view) + w * symmetric KL(p_self, p_ctx)

Each CE is the V4 masked 51-bin CE (query -> valid compartment -> cell mean)
against the same true center bins. The KL is averaged over valid queries and
is only a regularizer; the two CEs anchor both views to the data.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .subcellular_v4 import masked_bin_cross_entropy

LOSS_TYPE = 'two_view_masked_bin_ce'


def consistency_weight(cfg) -> float:
    specification = cfg.get('loss', {})
    if not isinstance(specification, dict) or set(specification) - {'type', 'consistency_weight'}:
        raise ValueError('V9 loss accepts only {type, consistency_weight}')
    if specification.get('type') != LOSS_TYPE:
        raise ValueError(f'V9 requires loss.type={LOSS_TYPE}')
    weight = specification.get('consistency_weight', .1)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError('loss.consistency_weight must be a nonnegative number')
    return float(weight)


def symmetric_kl(self_logits, ctx_logits, valid):
    if not bool(valid.any()):
        return self_logits[valid].float().sum()
    p = self_logits[valid].float().log_softmax(-1)
    q = ctx_logits[valid].float().log_softmax(-1)
    forward = F.kl_div(q, p, log_target=True, reduction='batchmean')   # KL(p || q)
    backward = F.kl_div(p, q, log_target=True, reduction='batchmean')  # KL(q || p)
    return .5 * (forward + backward)


def losses(output, batch, cfg, step):
    del step
    weight = consistency_weight(cfg)
    valid = batch['query_valid_mask'].bool()
    ce_self = masked_bin_cross_entropy(output['self_logits'], batch)
    ce_ctx = masked_bin_cross_entropy(output['ctx_logits'], batch)
    kl = symmetric_kl(output['self_logits'], output['ctx_logits'], valid)
    total = ce_self + ce_ctx + weight * kl
    return total, dict(total=float(total.detach()), ce_self=float(ce_self.detach()),
                       ce_ctx=float(ce_ctx.detach()), consistency=float(kl.detach()))
