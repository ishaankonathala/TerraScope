"""Explainable AI exports for ViT Attention Rollout, Swin Window Attention, and CNN Grad-CAM."""

from ml.explainability.attention_rollout import (
    INTERPRETABILITY_DISCLAIMER,
    compute_swin_window_attention,
    compute_vit_attention_rollout,
)
from ml.explainability.gradcam import compute_siamese_gradcam
from ml.explainability.visualization import (
    explain_prediction,
    overlay_heatmap_on_rgb,
    save_explainability_figure,
)

__all__ = [
    "INTERPRETABILITY_DISCLAIMER",
    "compute_siamese_gradcam",
    "compute_swin_window_attention",
    "compute_vit_attention_rollout",
    "explain_prediction",
    "overlay_heatmap_on_rgb",
    "save_explainability_figure",
]

