"""Model catalog and checkpoint availability endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_model_service
from backend.app.services.model_service import ModelService

router = APIRouter(tags=["Models"])


@router.get("/api/models")
def list_models(model_service: ModelService = Depends(get_model_service)) -> dict[str, Any]:
    return model_service.get_models_catalog()

