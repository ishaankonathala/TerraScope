"""Health-check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.api.dependencies import get_gee_config
from backend.app.core.config import get_settings
from backend.app.schemas.api_schemas import HealthResponse
from ml.satellite import get_gee_status
from ml.utils.device import device_info_dict

router = APIRouter(tags=["Health"])


@router.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    gee_status = get_gee_status(get_gee_config(settings)).to_dict()
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        device=device_info_dict(settings.device),
        gee_status=gee_status,
    )

