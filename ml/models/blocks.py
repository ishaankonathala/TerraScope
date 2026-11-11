"""Shared neural building blocks, temporal fusion layers, and base class for change detectors."""

from __future__ import annotations

import logging
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBNAct(nn.Module):
    """Standard 3x3 or 1x1 Convolution + BatchNorm + GELU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two-layer convolutional refinement block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, out_channels, kernel_size=3, padding=1),
            ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalFusionBlock(nn.Module):
    """Fuses bi-temporal feature maps via absolute difference and concatenation.

    Given ``f_before`` and ``f_after`` of shape ``(B, C, H, W)``, constructs
    ``[|f_before - f_after|, f_before, f_after]`` (3*C channels) and projects to
    ``out_channels`` so both explicit temporal discrepancy and semantic context
    are preserved.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            ConvBNAct(in_channels * 3, out_channels, kernel_size=1, padding=0),
            ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, f_before: torch.Tensor, f_after: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(f_before - f_after)
        fused = torch.cat([diff, f_before, f_after], dim=1)
        return self.proj(fused)


class DecoderUpBlock(nn.Module):
    """Bilinear upsampling + optional skip-connection fusion block."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.conv(x)


class BaseChangeDetector(nn.Module):
    """Unified interface for all bi-temporal change-detection architectures."""

    def __init__(
        self,
        model_name: str,
        architecture_family: str,
        backbone_name: str,
        in_channels: int = 3,
        num_classes: int = 1,
        img_size: int = 256,
        backbone_weights_source: str = "random_init",
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.architecture_family = architecture_family
        self.backbone_name = backbone_name
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone_weights_source = backbone_weights_source
        self.training_status = (
            "pretrained_backbone_untrained_head"
            if backbone_weights_source == "timm_pretrained"
            else "untrained"
        )

    def predict_proba(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """Compute pixel-wise change probabilities from temporal inputs."""
        logits = self.forward(before, after)
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)

    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "architecture_family": self.architecture_family,
            "backbone_name": self.backbone_name,
            "in_channels": self.in_channels,
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "backbone_weights_source": self.backbone_weights_source,
            "training_status": self.training_status,
            "total_parameters": self.parameter_count(),
            "trainable_parameters": self.trainable_parameter_count(),
        }


def create_timm_backbone(
    model_name: str,
    pretrained: bool = True,
    allow_random_fallback: bool = False,
    **kwargs: Any,
) -> tuple[nn.Module, str]:
    """Create a ``timm`` backbone and honestly record weight provenance."""
    if pretrained:
        try:
            model = timm.create_model(model_name, pretrained=True, **kwargs)
            return model, "timm_pretrained"
        except Exception as exc:
            if not allow_random_fallback:
                raise RuntimeError(
                    f"Failed to load pretrained timm weights for '{model_name}': {exc}"
                ) from exc
            logger.warning(
                "Could not load pretrained weights for '%s' (%s); falling back to random init.",
                model_name,
                exc,
            )

    model = timm.create_model(model_name, pretrained=False, **kwargs)
    return model, "random_init"

