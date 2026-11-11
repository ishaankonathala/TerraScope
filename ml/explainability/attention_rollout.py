"""Genuine Vision Transformer Attention Rollout and Swin Window-Attention extraction."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch

from ml.models.swin_change_detector import SwinChangeDetector
from ml.models.vit_change_detector import ViTChangeDetector

INTERPRETABILITY_DISCLAIMER = (
    "Explainability maps (Attention Rollout, Swin Window Attention, and Grad-CAM) are "
    "post-hoc interpretability aids showing spatial regions emphasized by internal features. "
    "They do not constitute definitive proof of causal reasoning."
)


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _rollout_from_attentions(
    attentions: list[torch.Tensor],
    num_prefix: int,
    grid_hw: tuple[int, int],
    target_hw: tuple[int, int],
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Compute Abnar & Zuidema (2020) attention rollout from per-layer attention matrices."""
    if not attentions:
        raise ValueError("No attention tensors supplied for attention rollout.")

    n_tokens = attentions[0].shape[-1]
    eye = np.eye(n_tokens, dtype=np.float32)
    rollout = eye.copy()

    for attn_tensor in attentions:
        # Take first batch item -> (num_heads, N, N)
        a = attn_tensor[0].numpy().astype(np.float32)
        if head_fusion == "max":
            fused = a.max(axis=0)
        elif head_fusion == "min":
            fused = a.min(axis=0)
        else:
            fused = a.mean(axis=0)

        if discard_ratio > 0.0:
            flat = fused.flatten()
            k = int(flat.size * discard_ratio)
            if k > 0:
                thresh = np.partition(flat, k)[k]
                fused = np.where(fused >= thresh, fused, 0.0)

        aug = 0.5 * fused + 0.5 * eye
        row_sums = aug.sum(axis=-1, keepdims=True)
        aug = aug / np.maximum(row_sums, 1e-8)
        rollout = aug @ rollout

    patch_rollout = rollout[num_prefix:, num_prefix:].mean(axis=0)
    if num_prefix > 0:
        cls_rollout = rollout[0, num_prefix:]
        spatial_vec = 0.5 * cls_rollout + 0.5 * patch_rollout
    else:
        spatial_vec = patch_rollout

    h_p, w_p = grid_hw
    if spatial_vec.size != h_p * w_p:
        raise ValueError(
            f"Patch vector length {spatial_vec.size} does not match grid {grid_hw}."
        )
    grid = spatial_vec.reshape(h_p, w_p)
    grid_norm = _normalize_map(grid)
    upsampled = cv2.resize(
        grid_norm, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC
    )
    return _normalize_map(upsampled)


def compute_vit_attention_rollout(
    model: ViTChangeDetector,
    before: torch.Tensor,
    after: torch.Tensor,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> dict[str, Any]:
    """Compute genuine Vision Transformer Attention Rollout for both temporal inputs."""
    if not isinstance(model, ViTChangeDetector):
        return {
            "supported": False,
            "method": "Attention Rollout (ViT)",
            "reason": "Provided model is not a ViTChangeDetector.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    model.eval()
    target_hw = (int(before.shape[2]), int(before.shape[3]))
    with torch.no_grad():
        logits, payload = model.forward_with_attention(before[:1], after[:1])

    if not payload.get("supported"):
        return {
            "supported": False,
            "method": "Attention Rollout (ViT)",
            "reason": "Backbone did not expose self-attention weight matrices.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    num_prefix = int(payload["num_prefix_tokens"])
    grid_hw = tuple(payload["patch_grid"])

    before_map = _rollout_from_attentions(
        payload["before_attns"],
        num_prefix=num_prefix,
        grid_hw=grid_hw,
        target_hw=target_hw,
        head_fusion=head_fusion,
        discard_ratio=discard_ratio,
    )
    after_map = _rollout_from_attentions(
        payload["after_attns"],
        num_prefix=num_prefix,
        grid_hw=grid_hw,
        target_hw=target_hw,
        head_fusion=head_fusion,
        discard_ratio=discard_ratio,
    )
    diff_map = _normalize_map(np.abs(after_map - before_map))
    prob_map = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

    return {
        "supported": True,
        "method": f"Attention Rollout (ViT, head_fusion={head_fusion})",
        "before_heatmap": before_map,
        "after_heatmap": after_map,
        "heatmap": _normalize_map(0.5 * diff_map + 0.5 * after_map),
        "diff_heatmap": diff_map,
        "prob_map": prob_map,
        "num_layers": len(payload["before_attns"]),
        "patch_grid": grid_hw,
        "disclaimer": INTERPRETABILITY_DISCLAIMER,
    }


def _reconstruct_swin_stage_maps(
    window_records: list[dict[str, Any]],
    stage_shapes: list[tuple[int, int]],
    target_hw: tuple[int, int],
) -> np.ndarray | None:
    """Reconstruct spatial attention saliency from Swin WindowAttention records."""
    stage_maps: list[np.ndarray] = []

    for rec in window_records:
        layer_idx = int(rec["layer"])
        if layer_idx >= len(stage_shapes):
            continue
        hs, ws = stage_shapes[layer_idx]
        wh, ww = rec["window_size"]
        attn = rec["attn"]  # (B*num_win, num_heads, M, M)

        h_pad = int(math.ceil(hs / wh) * wh)
        w_pad = int(math.ceil(ws / ww) * ww)
        expected_win = (h_pad // wh) * (w_pad // ww)

        if attn.shape[0] != expected_win or attn.shape[-1] != wh * ww:
            continue

        # Average over heads and query positions -> received attention per token in each window
        token_salience = attn.mean(dim=(1, 2)).numpy().astype(np.float32)  # (num_win, wh*ww)
        grid = token_salience.reshape(h_pad // wh, w_pad // ww, wh, ww)
        spatial = grid.transpose(0, 2, 1, 3).reshape(h_pad, w_pad)[:hs, :ws]
        spatial_norm = _normalize_map(spatial)
        up = cv2.resize(spatial_norm, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
        stage_maps.append(_normalize_map(up))

    if not stage_maps:
        return None
    return _normalize_map(np.mean(np.stack(stage_maps, axis=0), axis=0))


def compute_swin_window_attention(
    model: SwinChangeDetector,
    before: torch.Tensor,
    after: torch.Tensor,
) -> dict[str, Any]:
    """Compute spatial Window-Attention visualization for SwinChangeDetector."""
    if not isinstance(model, SwinChangeDetector):
        return {
            "supported": False,
            "method": "Swin Window Attention",
            "reason": "Provided model is not a SwinChangeDetector.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    model.eval()
    target_hw = (int(before.shape[2]), int(before.shape[3]))
    with torch.no_grad():
        logits, payload = model.forward_with_attention(before[:1], after[:1])

    if not payload.get("supported"):
        return {
            "supported": False,
            "method": "Swin Window Attention",
            "reason": "Backbone did not expose Swin WindowAttention weights.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    before_map = _reconstruct_swin_stage_maps(
        payload["before_window_attns"], payload["stage_shapes"], target_hw
    )
    after_map = _reconstruct_swin_stage_maps(
        payload["after_window_attns"], payload["stage_shapes"], target_hw
    )
    if before_map is None or after_map is None:
        return {
            "supported": False,
            "method": "Swin Window Attention",
            "reason": "Window attention dimensions could not be mapped cleanly to spatial stages.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    diff_map = _normalize_map(np.abs(after_map - before_map))
    prob_map = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

    return {
        "supported": True,
        "method": "Swin Shifted-Window Attention Aggregation",
        "before_heatmap": before_map,
        "after_heatmap": after_map,
        "heatmap": _normalize_map(0.5 * diff_map + 0.5 * after_map),
        "diff_heatmap": diff_map,
        "prob_map": prob_map,
        "num_attention_blocks": len(payload["before_window_attns"]),
        "stage_shapes": payload["stage_shapes"],
        "disclaimer": INTERPRETABILITY_DISCLAIMER,
    }

