"""Unified model factory and registry for change-detection architectures."""

from __future__ import annotations

from typing import Any

from ml.models.blocks import BaseChangeDetector
from ml.models.siamese_unet import SiameseUNet
from ml.models.swin_change_detector import SwinChangeDetector
from ml.models.vit_change_detector import ViTChangeDetector

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "siamese_unet": {
        "display_name": "Siamese U-Net (CNN Baseline)",
        "architecture_family": "CNN",
        "default_backbone": "resnet18",
        "supports_gradcam": True,
        "supports_attention": False,
        "description": (
            "Shared CNN encoder extracting multi-scale spatial features from both timestamps, "
            "fused via absolute difference + concatenation into a U-Net skip-connection decoder."
        ),
    },
    "vit": {
        "display_name": "Vision Transformer (Siamese ViT)",
        "architecture_family": "Vision Transformer",
        "default_backbone": "vit_tiny_patch16_224",
        "supports_gradcam": False,
        "supports_attention": True,
        "description": (
            "Shared ViT encoder reshaping multi-depth spatial patch tokens into 2D feature grids "
            "and decoding temporal discrepancies to full pixel resolution."
        ),
    },
    "swin": {
        "display_name": "Swin Transformer (Hierarchical Siamese)",
        "architecture_family": "Swin Transformer",
        "default_backbone": "swin_tiny_patch4_window7_224",
        "supports_gradcam": False,
        "supports_attention": True,
        "description": (
            "Shared hierarchical Swin Transformer backbone extracting 4-stage shifted-window "
            "representations fused across scales by an FPN/U-Net change decoder."
        ),
    },
}

_ALIASES: dict[str, str] = {
    "siamese_unet": "siamese_unet",
    "unet": "siamese_unet",
    "cnn": "siamese_unet",
    "vit": "vit",
    "vit_change_detector": "vit",
    "swin": "swin",
    "swin_change_detector": "swin",
    "swin_transformer": "swin",
}


def canonical_model_name(name: str) -> str:
    """Resolve supported model aliases to canonical key."""
    key = (name or "").strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown model name '{name}'. Supported models: {list(MODEL_REGISTRY.keys())}"
        )
    return _ALIASES[key]


def list_available_models() -> list[dict[str, Any]]:
    """Return catalog metadata for all registered change-detection architectures."""
    return [{"model_id": k, **v} for k, v in MODEL_REGISTRY.items()]


def get_model(
    name: str,
    in_channels: int = 3,
    num_classes: int = 1,
    img_size: int = 256,
    pretrained: bool = True,
    allow_random_fallback: bool = True,
    backbone_name: str | None = None,
    **kwargs: Any,
) -> BaseChangeDetector:
    """Instantiate a unified change-detection model by name.

    Example:
        >>> model = get_model("siamese_unet", pretrained=False)
        >>> model = get_model("vit", pretrained=False)
        >>> model = get_model("swin", pretrained=False)
    """
    canon = canonical_model_name(name)
    if canon == "siamese_unet":
        enc = backbone_name or MODEL_REGISTRY["siamese_unet"]["default_backbone"]
        return SiameseUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            encoder_name=enc,
            pretrained=pretrained,
            allow_random_fallback=allow_random_fallback,
            **kwargs,
        )
    if canon == "vit":
        bb = backbone_name or MODEL_REGISTRY["vit"]["default_backbone"]
        return ViTChangeDetector(
            in_channels=in_channels,
            num_classes=num_classes,
            img_size=img_size,
            backbone_name=bb,
            pretrained=pretrained,
            allow_random_fallback=allow_random_fallback,
            **kwargs,
        )
    bb = backbone_name or MODEL_REGISTRY["swin"]["default_backbone"]
    return SwinChangeDetector(
        in_channels=in_channels,
        num_classes=num_classes,
        img_size=img_size,
        backbone_name=bb,
        pretrained=pretrained,
        allow_random_fallback=allow_random_fallback,
        **kwargs,
    )

