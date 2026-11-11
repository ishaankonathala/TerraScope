"""Configurable Sentinel-2 cloud and cloud-shadow masking utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ml.preprocessing.image_loader import LoadedImage

# Sentinel-2 Scene Classification Layer (SCL) definitions:
# 0: No Data, 1: Saturated/Defective, 2: Dark Area Pixels, 3: Cloud Shadows,
# 4: Vegetation, 5: Bare Soils, 6: Water, 7: Unclassified,
# 8: Cloud Medium Probability, 9: Cloud High Probability, 10: Thin Cirrus, 11: Snow/Ice
DEFAULT_SCL_MASK_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10)


@dataclass
class CloudMaskConfig:
    """Configuration for pixel-level Sentinel-2 cloud and shadow masking."""

    method: str = "scl"  # "scl", "probability", "scl_and_prob", "none"
    scl_mask_classes: tuple[int, ...] = DEFAULT_SCL_MASK_CLASSES
    include_snow_ice: bool = False
    cloud_prob_threshold: float = 40.0

    def resolved_scl_classes(self) -> tuple[int, ...]:
        classes = set(self.scl_mask_classes)
        if self.include_snow_ice:
            classes.add(11)
        return tuple(sorted(classes))


@dataclass
class CloudMaskReport:
    """Summary of pixel-level cloud masking availability and coverage."""

    cloud_masking_available: bool
    method_used: str
    total_pixels: int
    valid_pixels: int
    masked_pixels: int
    cloud_shadow_fraction: float
    valid_fraction: float
    limitations_note: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extract_embedded_band(image: LoadedImage, candidate_names: set[str]) -> np.ndarray | None:
    """Locate an auxiliary quality band inside LoadedImage if named in metadata."""
    if not image.metadata.band_names:
        return None
    for idx, name in enumerate(image.metadata.band_names):
        if name.strip().upper() in candidate_names and idx < image.channels:
            return image.data[idx]
    return None


def apply_cloud_mask(
    image: LoadedImage,
    scl_band: np.ndarray | None = None,
    cloud_prob_band: np.ndarray | None = None,
    config: CloudMaskConfig | None = None,
) -> tuple[LoadedImage, CloudMaskReport]:
    """Apply configurable cloud/shadow masking and update ``image.valid_mask``.

    Masked pixels are marked False in ``valid_mask`` so downstream inference and
    evaluation exclude them rather than treating cloudy regions as unchanged.

    For standard RGB uploads lacking Sentinel-2 SCL or cloud-probability bands,
    this function leaves ``valid_mask`` intact and documents that reliable
    satellite cloud masking is unavailable.
    """
    cfg = config or CloudMaskConfig()
    total_pixels = int(image.height * image.width)

    if cfg.method.lower() == "none":
        valid_px = int(np.sum(image.valid_mask))
        return image, CloudMaskReport(
            cloud_masking_available=False,
            method_used="none",
            total_pixels=total_pixels,
            valid_pixels=valid_px,
            masked_pixels=total_pixels - valid_px,
            cloud_shadow_fraction=0.0,
            valid_fraction=valid_px / max(total_pixels, 1),
            limitations_note="Cloud masking disabled by configuration.",
        )

    scl = scl_band if scl_band is not None else _extract_embedded_band(image, {"SCL"})
    prob = (
        cloud_prob_band
        if cloud_prob_band is not None
        else _extract_embedded_band(image, {"MSK_CLDPRB", "CLOUD_PROBABILITY"})
    )

    if scl is None and prob is None:
        valid_px = int(np.sum(image.valid_mask))
        msg = (
            "Reliable satellite cloud and cloud-shadow masking is unavailable because "
            "this image lacks Sentinel-2 SCL or cloud-probability quality bands."
        )
        warnings = list(image.warnings)
        if msg not in warnings:
            warnings.append(msg)
        updated = LoadedImage(
            data=image.data,
            valid_mask=image.valid_mask.copy(),
            metadata=image.metadata,
            raw_dtype=image.raw_dtype,
            has_spectral_quality_bands=False,
            warnings=warnings,
        )
        return updated, CloudMaskReport(
            cloud_masking_available=False,
            method_used="unavailable_rgb_only",
            total_pixels=total_pixels,
            valid_pixels=valid_px,
            masked_pixels=total_pixels - valid_px,
            cloud_shadow_fraction=0.0,
            valid_fraction=valid_px / max(total_pixels, 1),
            limitations_note=msg,
            warnings=warnings,
        )

    expected_hw = (image.height, image.width)
    cloud_mask = np.zeros(expected_hw, dtype=bool)
    method_parts: list[str] = []

    if cfg.method in {"scl", "scl_and_prob"} and scl is not None:
        if scl.shape != expected_hw:
            raise ValueError(f"SCL band shape {scl.shape} does not match image {expected_hw}.")
        scl_int = np.rint(scl).astype(np.int32)
        for cls_id in cfg.resolved_scl_classes():
            cloud_mask |= scl_int == cls_id
        method_parts.append(f"SCL({','.join(str(c) for c in cfg.resolved_scl_classes())})")

    if cfg.method in {"probability", "scl_and_prob"} and prob is not None:
        if prob.shape != expected_hw:
            raise ValueError(
                f"Cloud probability band shape {prob.shape} does not match image {expected_hw}."
            )
        cloud_mask |= prob >= cfg.cloud_prob_threshold
        method_parts.append(f"CLDPRB>={cfg.cloud_prob_threshold:.0f}%")

    if not method_parts and scl is not None:
        scl_int = np.rint(scl).astype(np.int32)
        for cls_id in cfg.resolved_scl_classes():
            cloud_mask |= scl_int == cls_id
        method_parts.append("SCL_fallback")

    new_valid = image.valid_mask & (~cloud_mask)
    valid_px = int(np.sum(new_valid))
    masked_by_cloud = int(np.sum(cloud_mask))
    note = (
        "Pixel-level masking uses Sentinel-2 quality bands (SCL / cloud probability). "
        "Note: SCL classification has known edge-omission and bright-surface/terrain-shadow "
        "commission limitations; scene metadata cloud percentage is only an image-level filter."
    )
    updated = LoadedImage(
        data=image.data,
        valid_mask=new_valid,
        metadata=image.metadata,
        raw_dtype=image.raw_dtype,
        has_spectral_quality_bands=True,
        warnings=list(image.warnings),
    )
    return updated, CloudMaskReport(
        cloud_masking_available=True,
        method_used="+".join(method_parts),
        total_pixels=total_pixels,
        valid_pixels=valid_px,
        masked_pixels=total_pixels - valid_px,
        cloud_shadow_fraction=masked_by_cloud / max(total_pixels, 1),
        valid_fraction=valid_px / max(total_pixels, 1),
        limitations_note=note,
        warnings=list(image.warnings),
    )

