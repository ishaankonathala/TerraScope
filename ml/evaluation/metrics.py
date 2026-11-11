"""Segmentation and change-detection metrics with valid-mask and zero-division guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class ChangeDetectionMetrics:
    """Comprehensive binary pixel-level change-detection evaluation metrics."""

    tp: int
    fp: int
    fn: int
    tn: int
    total_valid_pixels: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    iou: float
    dice: float
    specificity: float
    macro_f1: float
    changed_pixel_ratio_pred: float
    changed_pixel_ratio_true: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def confusion_matrix(self) -> list[list[int]]:
        """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        return [[self.tn, self.fp], [self.fn, self.tp]]


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


class MetricAccumulator:
    """Streams confusion-matrix counts across batches without storing full arrays."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(
        self,
        preds: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        valid_mask: torch.Tensor | np.ndarray | None = None,
        from_logits: bool = True,
    ) -> None:
        if isinstance(preds, torch.Tensor):
            p = preds.detach()
            if from_logits:
                p = torch.sigmoid(p) if p.ndim < 4 or p.shape[1] == 1 else torch.softmax(p, dim=1)[:, 1:2]
            if p.ndim == 4 and p.shape[1] == 1:
                p = p.squeeze(1)
            p_np = p.cpu().numpy()
        else:
            p_np = np.asarray(preds, dtype=np.float32)
            if p_np.ndim == 4 and p_np.shape[1] == 1:
                p_np = p_np[:, 0]

        if isinstance(targets, torch.Tensor):
            t_np = targets.detach().cpu().numpy()
        else:
            t_np = np.asarray(targets)
        if t_np.ndim == 4 and t_np.shape[1] == 1:
            t_np = t_np[:, 0]

        pred_bin = p_np >= self.threshold
        true_bin = t_np > 0.5

        if valid_mask is not None:
            if isinstance(valid_mask, torch.Tensor):
                vm = valid_mask.detach().cpu().numpy().astype(bool)
            else:
                vm = np.asarray(valid_mask, dtype=bool)
            if vm.ndim == 4 and vm.shape[1] == 1:
                vm = vm[:, 0]
            pred_bin = pred_bin[vm]
            true_bin = true_bin[vm]

        self.tp += int(np.sum(pred_bin & true_bin))
        self.fp += int(np.sum(pred_bin & (~true_bin)))
        self.fn += int(np.sum((~pred_bin) & true_bin))
        self.tn += int(np.sum((~pred_bin) & (~true_bin)))

    def compute(self) -> ChangeDetectionMetrics:
        total = self.tp + self.fp + self.fn + self.tn
        accuracy = _safe_div(self.tp + self.tn, total)
        precision = _safe_div(self.tp, self.tp + self.fp)
        recall = _safe_div(self.tp, self.tp + self.fn)
        f1 = _safe_div(2 * self.tp, 2 * self.tp + self.fp + self.fn)
        iou = _safe_div(self.tp, self.tp + self.fp + self.fn)
        dice = f1
        specificity = _safe_div(self.tn, self.tn + self.fp)
        neg_f1 = _safe_div(2 * self.tn, 2 * self.tn + self.fp + self.fn)
        macro_f1 = 0.5 * (f1 + neg_f1)
        pred_ratio = _safe_div(self.tp + self.fp, total)
        true_ratio = _safe_div(self.tp + self.fn, total)
        return ChangeDetectionMetrics(
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            tn=self.tn,
            total_valid_pixels=total,
            accuracy=round(accuracy, 6),
            precision=round(precision, 6),
            recall=round(recall, 6),
            f1_score=round(f1, 6),
            iou=round(iou, 6),
            dice=round(dice, 6),
            specificity=round(specificity, 6),
            macro_f1=round(macro_f1, 6),
            changed_pixel_ratio_pred=round(pred_ratio, 6),
            changed_pixel_ratio_true=round(true_ratio, 6),
        )


def compute_binary_metrics(
    preds: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    valid_mask: torch.Tensor | np.ndarray | None = None,
    threshold: float = 0.5,
    from_logits: bool = False,
) -> ChangeDetectionMetrics:
    """Compute binary change-detection metrics directly from predicted probabilities or logits."""
    acc = MetricAccumulator(threshold=threshold)
    acc.update(preds=preds, targets=targets, valid_mask=valid_mask, from_logits=from_logits)
    return acc.compute()

