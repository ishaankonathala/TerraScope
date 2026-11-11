"""Loss functions for imbalanced binary and extensible multi-class change detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedBCEWithLogitsLoss(nn.Module):
    """Binary Cross-Entropy with logits supporting class imbalance weighting and valid masks."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        self.pos_weight_val = pos_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.ndim == 4 and logits.shape[1] == 1:
            logits_2d = logits.squeeze(1)
        else:
            logits_2d = logits
        targets_f = targets.float()

        pos_w = None
        if self.pos_weight_val is not None:
            pos_w = torch.tensor(self.pos_weight_val, dtype=logits_2d.dtype, device=logits_2d.device)

        loss_map = F.binary_cross_entropy_with_logits(
            logits_2d, targets_f, pos_weight=pos_w, reduction="none"
        )
        if valid_mask is not None:
            vm = valid_mask.float()
            denom = torch.clamp(vm.sum(), min=1.0)
            return (loss_map * vm).sum() / denom
        return loss_map.mean()


class DiceLoss(nn.Module):
    """Soft Dice loss over valid pixels for binary change segmentation."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits.squeeze(1) if logits.ndim == 4 and logits.shape[1] == 1 else logits)
        targets_f = targets.float()

        if valid_mask is not None:
            vm = valid_mask.float()
            probs = probs * vm
            targets_f = targets_f * vm

        probs_flat = probs.reshape(probs.shape[0], -1)
        targets_flat = targets_f.reshape(targets_f.shape[0], -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class CombinedBCEDiceLoss(nn.Module):
    """Weighted combination of BCE with logits and soft Dice loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss(smooth=smooth)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets, valid_mask) + self.dice_weight * self.dice(
            logits, targets, valid_mask
        )


class MultiClassChangeLoss(nn.Module):
    """Extensible cross-entropy loss for future multi-class land-use change labels."""

    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tgt = targets.long()
        if valid_mask is not None:
            tgt = tgt.clone()
            tgt[~valid_mask.bool()] = 255
        return self.ce(logits, tgt)


def get_loss_function(
    name: str = "bce_dice",
    pos_weight: float | None = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    num_classes: int = 1,
) -> nn.Module:
    """Instantiate a configured change-detection loss function."""
    key = (name or "bce_dice").strip().lower()
    if num_classes > 1 or key == "cross_entropy":
        return MultiClassChangeLoss()
    if key == "bce":
        return MaskedBCEWithLogitsLoss(pos_weight=pos_weight)
    if key == "dice":
        return DiceLoss()
    if key in {"bce_dice", "combined"}:
        return CombinedBCEDiceLoss(
            bce_weight=bce_weight, dice_weight=dice_weight, pos_weight=pos_weight
        )
    raise ValueError(f"Unsupported loss function '{name}'. Choose bce, dice, bce_dice, or cross_entropy.")

