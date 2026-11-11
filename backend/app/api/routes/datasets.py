"""Datasets inspection endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from backend.app.api.dependencies import get_dataset_service
from backend.app.services.dataset_service import DatasetService

router = APIRouter(tags=["Datasets"])


@router.get("/api/datasets")
def list_datasets(dataset_service: DatasetService = Depends(get_dataset_service)) -> dict[str, Any]:
    return dataset_service.list_datasets()

