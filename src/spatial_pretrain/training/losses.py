from __future__ import annotations

import torch
import torch.nn.functional as F

from spatial_pretrain.config import Config


def _masked_cross_entropy_per_token(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], targets[mask])


def _masked_cross_entropy_per_cell(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return logits.sum() * 0.0

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    mask_f = mask.to(dtype=per_token.dtype)
    masked_sum = (per_token * mask_f).sum(dim=-1)
    mask_count = mask_f.sum(dim=-1)
    valid_cells = mask_count > 0
    per_cell = masked_sum / mask_count.clamp_min(1.0)
    return per_cell[valid_cells].mean()


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    if reduction == "per_token":
        return _masked_cross_entropy_per_token(logits, targets, mask)
    if reduction == "per_cell":
        return _masked_cross_entropy_per_cell(logits, targets, mask)
    raise ValueError(f"Unsupported mask_reduction={reduction!r}")


def _select_mask_loss_inputs(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    mask_scope: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_logits = outputs["mask_logits"]
    targets = batch["target_bins"]
    loss_mask = batch["loss_mask"]

    if mask_scope == "all_cells":
        return mask_logits, targets, loss_mask
    if mask_scope == "center_cell":
        return mask_logits[:, 0], targets[:, 0], loss_mask[:, 0]
    raise ValueError(f"Unsupported mask_scope={mask_scope!r}")


def compute_pretraining_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: Config,
) -> dict[str, torch.Tensor]:
    mask_targets_logits, mask_targets, mask_loss_mask = _select_mask_loss_inputs(
        outputs,
        batch,
        config.loss.mask_scope,
    )

    mask_loss = _masked_cross_entropy(
        mask_targets_logits,
        mask_targets,
        mask_loss_mask,
        config.loss.mask_reduction,
    )
    return {
        "loss": mask_loss,
    }
