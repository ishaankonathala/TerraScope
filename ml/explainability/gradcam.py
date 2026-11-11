"""Grad-CAM spatial activation explanation for Siamese CNN change detection."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ml.explainability.attention_rollout import INTERPRETABILITY_DISCLAIMER, _normalize_map
from ml.models.siamese_unet import SiameseUNet


def compute_siamese_gradcam(
    model: SiameseUNet,
    before: torch.Tensor,
    after: torch.Tensor,
    target_layer: torch.nn.Module | None = None,
) -> dict[str, Any]:
    """Compute genuine Grad-CAM spatial importance map for SiameseUNet."""
    if not isinstance(model, SiameseUNet) and target_layer is None:
        return {
            "supported": False,
            "method": "Grad-CAM (CNN)",
            "reason": "Grad-CAM requires a SiameseUNet or explicit convolutional target layer.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    layer = target_layer or model.get_gradcam_target_layer()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _fwd_hook(_mod: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        activations.append(out)
        if out.requires_grad:
            out.register_hook(lambda grad: gradients.append(grad))

    handle = layer.register_forward_hook(_fwd_hook)
    prev_training = model.training
    model.eval()
    target_hw = (int(before.shape[2]), int(before.shape[3]))

    try:
        b_in = before[:1].detach().requires_grad_(True)
        a_in = after[:1].detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        logits = model(b_in, a_in)
        prob_map = torch.sigmoid(logits[0, 0]).detach().cpu().numpy()

        # Focus gradient signal on pixels predicted or tending toward change
        score = torch.sigmoid(logits).sum()
        score.backward()
    finally:
        handle.remove()
        model.train(prev_training)
        model.zero_grad(set_to_none=True)

    if not activations or not gradients:
        return {
            "supported": False,
            "method": "Grad-CAM (CNN)",
            "reason": "Target convolutional layer did not produce activations/gradients.",
            "disclaimer": INTERPRETABILITY_DISCLAIMER,
        }

    act = activations[0][0].detach().cpu()  # (C, H_f, W_f)
    grad = gradients[0][0].detach().cpu()  # (C, H_f, W_f)

    weights = grad.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)
    cam = torch.relu((weights * act).sum(dim=0)).numpy().astype(np.float32)
    cam_norm = _normalize_map(cam)
    cam_up = cv2.resize(cam_norm, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_CUBIC)
    cam_final = _normalize_map(cam_up)

    return {
        "supported": True,
        "method": "Grad-CAM (Siamese U-Net Decoder Features)",
        "heatmap": cam_final,
        "prob_map": prob_map,
        "disclaimer": INTERPRETABILITY_DISCLAIMER,
    }

