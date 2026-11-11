"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the FastAPI backend."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Satellite Land-Use Monitoring API"
    app_env: str = "development"
    log_level: str = "INFO"

    device: str = "auto"
    default_image_size: int = 256
    max_upload_size_mb: float = 50.0
    allow_pretrained_download: bool = True
    max_cached_models: int = 1

    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = Path("data")
    models_dir: Path = Path("models/checkpoints")
    experiments_dir: Path = Path("experiments")
    outputs_dir: Path = Path("outputs")
    upload_tmp_dir: Path = Path("outputs/tmp_uploads")

    gee_enabled: bool = False
    gee_project_id: str | None = None
    gee_service_account: str | None = None
    gee_private_key_path: str | None = None
    gee_export_destination: str = "local_geotiff"
    gee_drive_folder: str = "satellite_change_exports"
    gee_gcs_bucket: str | None = None

    def resolve_path(self, p: Path) -> Path:
        return p if p.is_absolute() else (self.project_root / p)

    @property
    def data_path(self) -> Path:
        return self.resolve_path(self.data_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve_path(self.models_dir)

    @property
    def experiments_path(self) -> Path:
        return self.resolve_path(self.experiments_dir)

    @property
    def outputs_path(self) -> Path:
        return self.resolve_path(self.outputs_dir)

    @property
    def upload_path(self) -> Path:
        return self.resolve_path(self.upload_tmp_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()

