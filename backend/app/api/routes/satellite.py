"""Google Earth Engine Sentinel-2 search, preview, export, and task status endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from backend.app.api.dependencies import get_gee_config, get_gee_registry
from backend.app.core.config import get_settings
from backend.app.schemas.api_schemas import (
    SatelliteExportRequest,
    SatellitePreviewRequest,
    SatelliteSearchRequest,
)
from backend.app.utils.files import validate_safe_id
from ml.satellite import (
    PRESET_REGIONS,
    AOISpec,
    GEETaskRegistry,
    build_temporal_composite,
    compute_gee_ndvi_preview,
    export_sentinel2_pair,
    get_export_task_status,
    get_gee_status,
    search_sentinel2_scenes,
)

router = APIRouter(tags=["Satellite / GEE"])


def _aoi_from_request(req: SatelliteSearchRequest) -> AOISpec:
    return AOISpec(
        bbox=req.bbox,
        geojson_geometry=req.geojson_geometry,
        place_name=req.place_name,
    )


@router.get("/api/satellite/status")
def satellite_status() -> dict[str, Any]:
    """Return Google Earth Engine configuration & authentication status."""
    cfg = get_gee_config(get_settings())
    status = get_gee_status(cfg).to_dict()
    status["preset_regions"] = PRESET_REGIONS
    return status


@router.post("/api/satellite/search")
def satellite_search(req: SatelliteSearchRequest) -> dict[str, Any]:
    """Search Sentinel-2 L2A (COPERNICUS/S2_SR_HARMONIZED) observations for before & after ranges."""
    cfg = get_gee_config(get_settings())
    try:
        aoi = _aoi_from_request(req)
        before_res = search_sentinel2_scenes(
            aoi=aoi,
            start_date=req.before_start,
            end_date=req.before_end,
            max_cloud_pct=req.max_cloud_pct,
            config=cfg,
        )
        after_res = search_sentinel2_scenes(
            aoi=aoi,
            start_date=req.after_start,
            end_date=req.after_end,
            max_cloud_pct=req.max_cloud_pct,
            config=cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not before_res.get("ok") and before_res.get("status_code") in {
        "disabled",
        "unauthenticated",
        "missing_project",
        "missing_library",
    }:
        raise HTTPException(status_code=503, detail=before_res.get("error", "GEE unavailable."))

    return {"before": before_res, "after": after_res}


@router.post("/api/satellite/preview")
def satellite_preview(req: SatellitePreviewRequest) -> dict[str, Any]:
    """Build masked Sentinel-2 temporal composites and NDVI comparison previews."""
    cfg = get_gee_config(get_settings())
    try:
        aoi = _aoi_from_request(req)
        comp_b = build_temporal_composite(
            aoi=aoi,
            start_date=req.before_start,
            end_date=req.before_end,
            bands=req.bands,
            max_cloud_pct=req.max_cloud_pct,
            scale_m=req.scale_m,
            target_crs=req.target_crs,
            apply_scl_mask=req.apply_scl_mask,
            config=cfg,
        )
        comp_a = build_temporal_composite(
            aoi=aoi,
            start_date=req.after_start,
            end_date=req.after_end,
            bands=req.bands,
            max_cloud_pct=req.max_cloud_pct,
            scale_m=req.scale_m,
            target_crs=req.target_crs,
            apply_scl_mask=req.apply_scl_mask,
            config=cfg,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for comp in (comp_b, comp_a):
        if not comp.get("ok"):
            code = 503 if comp.get("status_code") in {"disabled", "unauthenticated", "missing_project"} else 400
            raise HTTPException(status_code=code, detail=comp.get("error", "Composite failed."))

    ndvi_preview = None
    if "B4" in req.bands and "B8" in req.bands:
        ndvi_preview = compute_gee_ndvi_preview(
            comp_b["ee_image"], comp_a["ee_image"], comp_b["ee_geometry"], scale_m=max(req.scale_m, 20)
        )

    return {
        "before": {"thumb_url": comp_b["thumb_url"], "metadata": comp_b["metadata"]},
        "after": {"thumb_url": comp_a["thumb_url"], "metadata": comp_a["metadata"]},
        "ndvi": ndvi_preview,
    }


@router.post("/api/satellite/export")
def satellite_export(
    req: SatelliteExportRequest,
    registry: GEETaskRegistry = Depends(get_gee_registry),
) -> dict[str, Any]:
    """Export aligned Sentinel-2 before/after GeoTIFFs locally or to Drive/GCS."""
    settings = get_settings()
    cfg = get_gee_config(settings)
    try:
        aoi = _aoi_from_request(req)
        res = export_sentinel2_pair(
            aoi=aoi,
            before_start=req.before_start,
            before_end=req.before_end,
            after_start=req.after_start,
            after_end=req.after_end,
            bands=req.bands,
            max_cloud_pct=req.max_cloud_pct,
            scale_m=req.scale_m,
            target_crs=req.target_crs,
            destination=req.destination,
            output_dir=settings.data_path / "samples" / "gee_exports",
            config=cfg,
            registry=registry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not res.get("ok"):
        code = 503 if res.get("status_code") in {"disabled", "unauthenticated", "missing_project"} else 400
        raise HTTPException(status_code=code, detail=res.get("error", "Export failed."))
    return res


@router.get("/api/satellite/tasks/{task_id}")
def satellite_task_status(
    task_id: str,
    registry: GEETaskRegistry = Depends(get_gee_registry),
) -> dict[str, Any]:
    """Poll status of a tracked Earth Engine export task."""
    safe_id = validate_safe_id(task_id, "task_id")
    status = get_export_task_status(safe_id, registry=registry)
    if not status.get("found"):
        raise HTTPException(status_code=404, detail=status.get("error", "Task not found."))
    return status

