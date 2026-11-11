"""Vegetation index (NDVI) calculation and comparative indicators with safety guards."""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.preprocessing.image_loader import LoadedImage

NDVI_SCIENTIFIC_NOTE = (
    "NDVI differences reflect relative changes in spectral red/NIR vegetation vigor. "
    "An NDVI decrease alone does not prove deforestation, wildfire, or construction "
    "because seasonal phenology, moisture stress, and crop harvesting also alter NDVI."
)


def compute_ndvi(
    red_b4: np.ndarray,
    nir_b8: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compute ``NDVI = (B8 - B4) / (B8 + B4)`` with zero-denominator and invalid guards."""
    red = red_b4.astype(np.float32)
    nir = nir_b8.astype(np.float32)
    denom = nir + red

    valid = np.isfinite(red) & np.isfinite(nir) & (np.abs(denom) > 1e-6)
    if valid_mask is not None:
        valid &= valid_mask.astype(bool)

    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = np.clip((nir[valid] - red[valid]) / denom[valid], -1.0, 1.0)

    vals = ndvi[valid]
    total_px = int(red.size)
    valid_px = int(vals.size)
    stats = {
        "valid_pixels": valid_px,
        "total_pixels": total_px,
        "valid_coverage_pct": round((valid_px / max(total_px, 1)) * 100.0, 3),
        "mean": round(float(np.mean(vals)), 4) if valid_px else None,
        "median": round(float(np.median(vals)), 4) if valid_px else None,
        "std": round(float(np.std(vals)), 4) if valid_px else None,
        "min": round(float(np.min(vals)), 4) if valid_px else None,
        "max": round(float(np.max(vals)), 4) if valid_px else None,
    }
    return ndvi, stats


def compare_ndvi(
    before_b4: np.ndarray,
    before_b8: np.ndarray,
    after_b4: np.ndarray,
    after_b8: np.ndarray,
    before_valid: np.ndarray | None = None,
    after_valid: np.ndarray | None = None,
    delta_threshold: float = 0.15,
) -> dict[str, Any]:
    """Compute before/after NDVI maps, difference map, valid coverage, and summary stats."""
    ndvi_before, stats_before = compute_ndvi(before_b4, before_b8, valid_mask=before_valid)
    ndvi_after, stats_after = compute_ndvi(after_b4, after_b8, valid_mask=after_valid)

    joint_valid = np.isfinite(ndvi_before) & np.isfinite(ndvi_after)
    ndvi_diff = np.full(ndvi_before.shape, np.nan, dtype=np.float32)
    ndvi_diff[joint_valid] = ndvi_after[joint_valid] - ndvi_before[joint_valid]

    diff_vals = ndvi_diff[joint_valid]
    n_joint = int(diff_vals.size)
    total_px = int(ndvi_before.size)

    if n_joint > 0:
        decrease_pct = float(np.sum(diff_vals <= -abs(delta_threshold))) / n_joint * 100.0
        increase_pct = float(np.sum(diff_vals >= abs(delta_threshold))) / n_joint * 100.0
        diff_stats = {
            "mean_diff": round(float(np.mean(diff_vals)), 4),
            "median_diff": round(float(np.median(diff_vals)), 4),
            "std_diff": round(float(np.std(diff_vals)), 4),
            "p10_diff": round(float(np.percentile(diff_vals, 10)), 4),
            "p90_diff": round(float(np.percentile(diff_vals, 90)), 4),
            "ndvi_decrease_indicator_pct": round(decrease_pct, 3),
            "ndvi_increase_indicator_pct": round(increase_pct, 3),
            "delta_threshold": delta_threshold,
        }
    else:
        diff_stats = {
            "mean_diff": None,
            "median_diff": None,
            "std_diff": None,
            "p10_diff": None,
            "p90_diff": None,
            "ndvi_decrease_indicator_pct": None,
            "ndvi_increase_indicator_pct": None,
            "delta_threshold": delta_threshold,
        }

    return {
        "available": True,
        "ndvi_before": ndvi_before,
        "ndvi_after": ndvi_after,
        "ndvi_diff": ndvi_diff,
        "joint_valid_mask": joint_valid,
        "valid_coverage_pct": round((n_joint / max(total_px, 1)) * 100.0, 3),
        "before_stats": stats_before,
        "after_stats": stats_after,
        "diff_stats": diff_stats,
        "scientific_note": NDVI_SCIENTIFIC_NOTE,
    }


def _find_band_index(band_names: list[str] | None, target_aliases: set[str]) -> int | None:
    if not band_names:
        return None
    for idx, name in enumerate(band_names):
        if name.strip().upper() in target_aliases:
            return idx
    return None


def compute_ndvi_from_loaded_pair(
    before: LoadedImage,
    after: LoadedImage,
) -> dict[str, Any]:
    """Extract B4 (Red) and B8 (NIR) from LoadedImage pair when present and compare NDVI."""
    b_red_idx = _find_band_index(before.metadata.band_names, {"B4", "B04", "RED"})
    b_nir_idx = _find_band_index(before.metadata.band_names, {"B8", "B08", "NIR"})
    a_red_idx = _find_band_index(after.metadata.band_names, {"B4", "B04", "RED"})
    a_nir_idx = _find_band_index(after.metadata.band_names, {"B8", "B08", "NIR"})

    if None in (b_red_idx, b_nir_idx, a_red_idx, a_nir_idx):
        return {
            "available": False,
            "reason": (
                "NDVI requires both Red (B4) and Near-Infrared (B8) spectral bands; "
                "one or both bands are missing from the loaded imagery."
            ),
            "scientific_note": NDVI_SCIENTIFIC_NOTE,
        }

    return compare_ndvi(
        before_b4=before.data[b_red_idx],
        before_b8=before.data[b_nir_idx],
        after_b4=after.data[a_red_idx],
        after_b8=after.data[a_nir_idx],
        before_valid=before.valid_mask,
        after_valid=after.valid_mask,
    )


def compute_gee_ndvi_preview(
    before_ee_img: Any,
    after_ee_img: Any,
    ee_geometry: Any,
    scale_m: int = 20,
) -> dict[str, Any]:
    """Compute server-side NDVI thumbnails and statistics in Google Earth Engine."""
    try:
        import ee

        def _ndvi_img(img: Any) -> Any:
            return img.normalizedDifference(["B8", "B4"]).rename("NDVI")

        b_ndvi = _ndvi_img(before_ee_img)
        a_ndvi = _ndvi_img(after_ee_img)
        diff_ndvi = a_ndvi.subtract(b_ndvi).rename("NDVI_DIFF")

        stats_b = b_ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=ee_geometry,
            scale=scale_m,
            maxPixels=1_000_000,
        ).getInfo()
        stats_a = a_ndvi.reduceRegion(
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
            geometry=ee_geometry,
            scale=scale_m,
            maxPixels=1_000_000,
        ).getInfo()

        b_url = b_ndvi.getThumbURL(
            {
                "min": -0.2,
                "max": 0.8,
                "palette": ["#8c510a", "#f6e8c3", "#a1d99b", "#006837"],
                "dimensions": 512,
                "region": ee_geometry,
                "format": "png",
            }
        )
        a_url = a_ndvi.getThumbURL(
            {
                "min": -0.2,
                "max": 0.8,
                "palette": ["#8c510a", "#f6e8c3", "#a1d99b", "#006837"],
                "dimensions": 512,
                "region": ee_geometry,
                "format": "png",
            }
        )
        diff_url = diff_ndvi.getThumbURL(
            {
                "min": -0.5,
                "max": 0.5,
                "palette": ["#b2182b", "#f7f7f7", "#1b7837"],
                "dimensions": 512,
                "region": ee_geometry,
                "format": "png",
            }
        )
        return {
            "available": True,
            "before_ndvi_url": b_url,
            "after_ndvi_url": a_url,
            "diff_ndvi_url": diff_url,
            "before_stats": stats_b,
            "after_stats": stats_a,
            "scientific_note": NDVI_SCIENTIFIC_NOTE,
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Could not compute GEE NDVI preview: {exc}",
            "scientific_note": NDVI_SCIENTIFIC_NOTE,
        }

