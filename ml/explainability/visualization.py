"""Explainability overlays, comparative figures, and unified dispatcher."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.explainability.attention_rollout import (
    INTERPRETABILITY_DISCLAIMER,
    compute_swin_window_attention,
    compute_vit_attention_rollout,
)
from ml.explainability.gradcam import compute_siamese_gradcam
from ml.models.blocks import BaseChangeDetector
from ml.models.siamese_unet import SiameseUNet
from ml.models.swin_change_detector import SwinChangeDetector
from ml.models.vit_change_detector import ViTChangeDetector


def overlay_heatmap_on_rgb(
    rgb_uint8: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Blend a normalized ``[0, 1]`` heatmap onto an ``(H, W, 3)`` uint8 image."""
    hm_u8 = np.clip(np.round(heatmap * 255.0), 0, 255).astype(np.uint8)
    colored_bgr = cv2.applyColorMap(hm_u8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = rgb_uint8.astype(np.float32)
    blended = (1.0 - alpha) * base + alpha * colored_rgb
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def save_explainability_figure(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    heatmap: np.ndarray,
    prob_map: np.ndarray,
    output_path: str | Path,
    method_title: str,
) -> Path:
    """Save a 4-panel explainability comparison figure."""
    overlay = overlay_heatmap_on_rgb(after_rgb, heatmap, alpha=0.48)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=140)

    axes[0].imshow(before_rgb)
    axes[0].set_title("Before (T1)")
    axes[1].imshow(after_rgb)
    axes[1].set_title("After (T2)")
    axes[2].imshow(overlay)
    axes[2].set_title(method_title)
    axes[3].imshow(prob_map, cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("Predicted Change Prob.")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"Explainability Inspection — {method_title}", fontsize=11)
    fig.text(0.5, 0.01, INTERPRETABILITY_DISCLAIMER, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def explain_prediction(
    model: BaseChangeDetector,
    before_tensor: torch.Tensor,
    after_tensor: torch.Tensor,
    before_rgb: np.ndarray | None = None,
    after_rgb: np.ndarray | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Dispatch to the genuine architecture-appropriate XAI method."""
    if isinstance(model, ViTChangeDetector):
        result = compute_vit_attention_rollout(model, before_tensor, after_tensor)
    elif isinstance(model, SwinChangeDetector):
        result = compute_swin_window_attention(model, before_tensor, after_tensor)
    elif isinstance(model, SiameseUNet):
        result = compute_siamese_gradcam(model, before_tensor, after_tensor)
    else:
        return {
            "supported": False,
            "method": "Unavailable",
            "reason": f"No explainability adapter registered for {type(model).__name__}.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    if not result.get("supported"):
        return result

    if after_rgb is not None:
        result["overlay_rgb"] = overlay_heatmap_on_rgb(after_rgb, result["heatmap"])
        if before_rgb is not None and "before_heatmap" in result:
            result["before_overlay_rgb"] = overlay_heatmap_on_rgb(
                before_rgb, result["before_heatmap"]
            )
            result["after_overlay_rgb"] = overlay_heatmap_on_rgb(
                after_rgb, result["after_heatmap"]
            )
        if output_path is not None and before_rgb is not None:
            saved = save_explainability_figure(
                before_rgb=before_rgb,
                after_rgb=after_rgb,
                heatmap=result["heatmap"],
                prob_map=result["prob_map"],
                output_path=output_path,
                method_title=result["method"],
            )
            result["figure_path"] = str(saved)

    return result

