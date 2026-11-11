"""Change-detection model architectures and factory exports."""

from ml.models.blocks import BaseChangeDetector, TemporalFusionBlock
from ml.models.model_factory import (
    MODEL_REGISTRY,
    canonical_model_name,
    get_model,
    list_available_models,
)
from ml.models.siamese_unet import SiameseUNet
from ml.models.swin_change_detector import SwinChangeDetector
from ml.models.vit_change_detector import ViTChangeDetector

__all__ = [
    "BaseChangeDetector",
    "MODEL_REGISTRY",
    "SiameseUNet",
    "SwinChangeDetector",
    "TemporalFusionBlock",
    "ViTChangeDetector",
    "canonical_model_name",
    "get_model",
    "list_available_models",
]

