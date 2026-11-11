"""Shared FastAPI service dependency providers."""

from __future__ import annotations

from functools import lru_cache

from backend.app.core.config import Settings, get_settings
from backend.app.services.dataset_service import DatasetService
from backend.app.services.inference_service import InferenceService
from backend.app.services.model_service import ModelService
from ml.satellite import GEEConfig, GEETaskRegistry


@lru_cache
def get_model_service() -> ModelService:
    return ModelService(get_settings())


@lru_cache
def get_inference_service() -> InferenceService:
    return InferenceService(get_settings(), get_model_service())


@lru_cache
def get_dataset_service() -> DatasetService:
    return DatasetService(get_settings())


@lru_cache
def get_gee_registry() -> GEETaskRegistry:
    settings = get_settings()
    return GEETaskRegistry(state_dir=settings.outputs_path / "gee_exports")


def get_gee_config(settings: Settings | None = None) -> GEEConfig:
    s = settings or get_settings()
    return GEEConfig(
        enabled=s.gee_enabled,
        project_id=s.gee_project_id,
        service_account=s.gee_service_account,
        private_key_path=s.gee_private_key_path,
        export_destination=s.gee_export_destination,
        drive_folder=s.gee_drive_folder,
        gcs_bucket=s.gee_gcs_bucket,
    )

