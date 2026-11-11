"""Configurable post-processing and connected-component region analysis for change masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class PostprocessingConfig:
    """Configurable morphological and connected-component post-processing options."""

    enabled: bool = False
    morph_operation: str = "none"  # "none", "open", "close", "open_close"
    morph_kernel_size: int = 3
    min_region_pixels: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_connected_regions(binary_mask: np.ndarray) -> list[dict[str, Any]]:
    """Extract connected changed components with bounding boxes and centroids."""
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    regions: list[dict[str, Any]] = []
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area_px = int(stats[label_id, cv2.CC_STAT_AREA])
        cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])
        regions.append(
            {
                "region_id": label_id,
                "pixel_count": area_px,
                "bbox": [x, y, w, h],
                "centroid": [round(cx, 2), round(cy, 2)],
            }
        )
    regions.sort(key=lambda r: r["pixel_count"], reverse=True)
    return regions


def postprocess_mask(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    config: PostprocessingConfig | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply optional morphological cleaning and small-region removal.

    By default ``config.enabled`` is False so raw model predictions are preserved
    faithfully without concealing errors. Valid-pixel masking is always enforced.
    """
    cfg = config or PostprocessingConfig()
    work = (binary_mask > 0).astype(np.uint8)
    if valid_mask is not None:
        work = work & valid_mask.astype(np.uint8)

    raw_pixels = int(np.sum(work))
    removed_small = 0

    if cfg.enabled:
        k = max(int(cfg.morph_kernel_size), 1)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        op = (cfg.morph_operation or "none").strip().lower()
        if op == "open":
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
        elif op == "open_close":
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
        elif op != "none":
            raise ValueError(
                f"Unsupported morph_operation '{cfg.morph_operation}'. Choose none, open, close, open_close."
            )

        if cfg.min_region_pixels > 0:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
            filtered = np.zeros_like(work)
            for lbl in range(1, num_labels):
                if int(stats[lbl, cv2.CC_STAT_AREA]) >= cfg.min_region_pixels:
                    filtered[labels == lbl] = 1
                else:
                    removed_small += 1
            work = filtered

        if valid_mask is not None:
            work = work & valid_mask.astype(np.uint8)

    regions = extract_connected_regions(work)
    summary = {
        "enabled": cfg.enabled,
        "morph_operation": cfg.morph_operation if cfg.enabled else "none",
        "min_region_pixels": cfg.min_region_pixels if cfg.enabled else 0,
        "raw_changed_pixels": raw_pixels,
        "postprocessed_changed_pixels": int(np.sum(work)),
        "removed_small_components": removed_small,
        "num_regions": len(regions),
    }
    return work, {"summary": summary, "regions": regions}

