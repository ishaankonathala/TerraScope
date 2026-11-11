"""Sentinel-2 temporal composite construction with unified grid and projection enforcement."""

from __future__ import annotations

from typing import Any, Sequence

from ml.satellite.gee_client import GEEConfig, initialize_gee
from ml.satellite.sentinel2 import (
    SENTINEL2_BANDS,
    SENTINEL2_COLLECTION,
    AOISpec,
    mask_s2_scl_ee,
    validate_sentinel2_params,
)

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]


def build_temporal_composite(
    aoi: AOISpec,
    start_date: str,
    end_date: str,
    bands: Sequence[str] = ("B4", "B3", "B2", "B8"),
    max_cloud_pct: float = 20.0,
    scale_m: int = 10,
    target_crs: str = "EPSG:4326",
    apply_scl_mask: bool = True,
    config: GEEConfig | None = None,
) -> dict[str, Any]:
    """Build a pixel-masked Sentinel-2 median composite locked to an explicit CRS and scale.

    Returns both the server-side ``ee.Image`` (for export/preview) and factual
    provenance metadata (scene count, acquisition dates, band native scales, and
    resampling notes when mixing 10m and 20m bands).
    """
    clean_bands, clean_scale = validate_sentinel2_params(
        bands, start_date, end_date, max_cloud_pct, scale_m
    )
    status = initialize_gee(config, force=True)
    if not status.initialized:
        return {"ok": False, "status_code": status.status_code, "error": status.message}

    try:
        geom = aoi.to_ee_geometry()
        raw_col = (
            ee.ImageCollection(SENTINEL2_COLLECTION)
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", float(max_cloud_pct)))
        )

        scene_count = int(raw_col.size().getInfo())
        if scene_count == 0:
            return {
                "ok": False,
                "status_code": "empty_collection",
                "error": (
                    f"No valid Sentinel-2 observations found for {start_date}..{end_date} "
                    f"with cloud <= {max_cloud_pct}%."
                ),
            }

        if apply_scl_mask:
            masked_col = raw_col.map(lambda img: mask_s2_scl_ee(img))
        else:
            masked_col = raw_col

        export_bands = [b for b in clean_bands if b != "SCL"]
        if not export_bands:
            export_bands = ["B4", "B3", "B2"]

        composite = (
            masked_col.select(export_bands)
            .median()
            .clip(geom)
            .reproject(crs=target_crs, scale=clean_scale)
            .toFloat()
        )

        # Generate a quick-look RGB thumbnail URL (B4, B3, B2) when available
        vis_bands = (
            ["B4", "B3", "B2"]
            if all(b in export_bands for b in ("B4", "B3", "B2"))
            else export_bands[:3]
        )
        thumb_url = composite.select(vis_bands).getThumbURL(
            {"min": 0, "max": 3000, "dimensions": 512, "region": geom, "format": "png"}
        )

        native_scales = {b: SENTINEL2_BANDS[b]["native_scale_m"] for b in export_bands}
        mixed_res_note = (
            f"Bands resampled onto unified {target_crs} grid at {clean_scale}m scale "
            f"(native scales: {native_scales})."
        )

        return {
            "ok": True,
            "ee_image": composite,
            "ee_geometry": geom,
            "thumb_url": thumb_url,
            "metadata": {
                "collection": SENTINEL2_COLLECTION,
                "start_date": start_date,
                "end_date": end_date,
                "scene_count": scene_count,
                "bands": export_bands,
                "native_band_scales_m": native_scales,
                "max_cloud_pct_filter": max_cloud_pct,
                "scl_pixel_masking": apply_scl_mask,
                "target_crs": target_crs,
                "target_scale_m": clean_scale,
                "aoi_bounds": aoi.resolve_bounds(),
                "resampling_note": mixed_res_note,
            },
        }
    except Exception as exc:
        return {
            "ok": False,
            "status_code": "composite_error",
            "error": f"Failed to construct Sentinel-2 composite: {exc}",
        }

