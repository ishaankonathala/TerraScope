"""Pydantic v2 request and response schemas for the REST API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    device: dict[str, Any]
    gee_status: dict[str, Any]


class UploadPairResponse(BaseModel):
    upload_id: str
    before_filename: str
    after_filename: str
    before_metadata: dict[str, Any]
    after_metadata: dict[str, Any]
    pair_validation: dict[str, Any]


class PredictByUploadRequest(BaseModel):
    upload_id: str
    model_name: Literal["siamese_unet", "vit", "swin"] = "siamese_unet"
    checkpoint_name: str | None = None
    threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    tile_size: int = Field(default=256, ge=64, le=1024)
    stride: int = Field(default=192, ge=32, le=1024)
    normalize_mode: Literal["imagenet", "sentinel2_sr", "unit", "zscore", "percentile"] = "imagenet"
    postprocess_enabled: bool = False
    morph_operation: Literal["none", "open", "close", "open_close"] = "none"
    min_region_pixels: int = Field(default=0, ge=0, le=10000)
    include_xai: bool = False


class PredictionResponse(BaseModel):
    result_id: str
    model_name: str
    training_status: str
    threshold: float
    changed_pixels: int
    total_valid_pixels: int
    masked_invalid_pixels: int
    change_percentage: float
    inference_time_sec: float
    geospatial: dict[str, Any]
    regions: list[dict[str, Any]]
    artifacts: dict[str, str]
    explainability: dict[str, Any] | None = None
    metadata: dict[str, Any]


class EvaluatePairRequest(BaseModel):
    result_id: str
    ground_truth_upload_id: str | None = None


class SatelliteSearchRequest(BaseModel):
    bbox: tuple[float, float, float, float] | None = None
    place_name: str | None = None
    geojson_geometry: dict[str, Any] | None = None
    before_start: str
    before_end: str
    after_start: str
    after_end: str
    max_cloud_pct: float = Field(default=20.0, ge=0.0, le=100.0)


class SatellitePreviewRequest(SatelliteSearchRequest):
    bands: list[str] = Field(default_factory=lambda: ["B4", "B3", "B2", "B8"])
    scale_m: int = Field(default=20, ge=10, le=250)
    target_crs: str = "EPSG:4326"
    apply_scl_mask: bool = True


class SatelliteExportRequest(SatellitePreviewRequest):
    destination: Literal["local_geotiff", "google_drive", "gcs"] = "local_geotiff"

