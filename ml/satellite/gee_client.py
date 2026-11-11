"""Google Earth Engine authentication, initialization, and status management."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]


@dataclass
class GEEConfig:
    """Environment-driven Google Earth Engine configuration (no hardcoded secrets)."""

    enabled: bool = False
    project_id: str | None = None
    service_account: str | None = None
    private_key_path: str | None = None
    export_destination: str = "local_geotiff"
    drive_folder: str = "satellite_change_exports"
    gcs_bucket: str | None = None

    @classmethod
    def from_env(cls) -> "GEEConfig":
        enabled_raw = os.getenv("GEE_ENABLED", "false").strip().lower()
        return cls(
            enabled=enabled_raw in {"1", "true", "yes", "on"},
            project_id=os.getenv("GEE_PROJECT_ID", "").strip() or None,
            service_account=os.getenv("GEE_SERVICE_ACCOUNT", "").strip() or None,
            private_key_path=os.getenv("GEE_PRIVATE_KEY_PATH", "").strip() or None,
            export_destination=os.getenv("GEE_EXPORT_DESTINATION", "local_geotiff").strip()
            or "local_geotiff",
            drive_folder=os.getenv("GEE_DRIVE_FOLDER", "satellite_change_exports").strip()
            or "satellite_change_exports",
            gcs_bucket=os.getenv("GEE_GCS_BUCKET", "").strip() or None,
        )


@dataclass
class GEEStatus:
    """Factual runtime status of Google Earth Engine availability."""

    library_installed: bool
    enabled: bool
    initialized: bool
    status_code: str  # ready, disabled, unauthenticated, missing_project, missing_library, error
    auth_mode: str | None
    project_id: str | None
    export_destination: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _interactive_credentials_exist() -> bool:
    cred_path = Path.home() / ".config" / "earthengine" / "credentials"
    return cred_path.is_file() and cred_path.stat().st_size > 0


def initialize_gee(config: GEEConfig | None = None, force: bool = False) -> GEEStatus:
    """Initialize the official Earth Engine Python API safely without hardcoded credentials."""
    cfg = config or GEEConfig.from_env()

    if ee is None:
        return GEEStatus(
            library_installed=False,
            enabled=cfg.enabled,
            initialized=False,
            status_code="missing_library",
            auth_mode=None,
            project_id=cfg.project_id,
            export_destination=cfg.export_destination,
            message="earthengine-api is not installed. Install with `pip install earthengine-api`.",
        )

    if not cfg.enabled and not force:
        return GEEStatus(
            library_installed=True,
            enabled=False,
            initialized=False,
            status_code="disabled",
            auth_mode=None,
            project_id=cfg.project_id,
            export_destination=cfg.export_destination,
            message=(
                "Google Earth Engine integration is disabled (GEE_ENABLED=false). "
                "Local uploads and benchmark datasets remain fully operational."
            ),
        )

    # Check if already initialized in current process
    if getattr(ee.data, "_initialized", False):
        return GEEStatus(
            library_installed=True,
            enabled=True,
            initialized=True,
            status_code="ready",
            auth_mode="service_account" if cfg.service_account else "interactive_oauth",
            project_id=cfg.project_id,
            export_destination=cfg.export_destination,
            message="Google Earth Engine session is initialized and ready.",
        )

    if cfg.service_account or cfg.private_key_path:
        if not (cfg.service_account and cfg.private_key_path):
            return GEEStatus(
                library_installed=True,
                enabled=True,
                initialized=False,
                status_code="unauthenticated",
                auth_mode="service_account",
                project_id=cfg.project_id,
                export_destination=cfg.export_destination,
                message=(
                    "Both GEE_SERVICE_ACCOUNT and GEE_PRIVATE_KEY_PATH must be set for "
                    "service-account authentication."
                ),
            )
        key_file = Path(cfg.private_key_path)
        if not key_file.is_file():
            return GEEStatus(
                library_installed=True,
                enabled=True,
                initialized=False,
                status_code="unauthenticated",
                auth_mode="service_account",
                project_id=cfg.project_id,
                export_destination=cfg.export_destination,
                message=f"Service account key file not found at configured path: {key_file}",
            )
        try:
            credentials = ee.ServiceAccountCredentials(cfg.service_account, str(key_file))
            ee.Initialize(credentials=credentials, project=cfg.project_id)
            return GEEStatus(
                library_installed=True,
                enabled=True,
                initialized=True,
                status_code="ready",
                auth_mode="service_account",
                project_id=cfg.project_id,
                export_destination=cfg.export_destination,
                message="Initialized Google Earth Engine via service account.",
            )
        except Exception as exc:
            return GEEStatus(
                library_installed=True,
                enabled=True,
                initialized=False,
                status_code="error",
                auth_mode="service_account",
                project_id=cfg.project_id,
                export_destination=cfg.export_destination,
                message=f"Earth Engine service-account initialization failed: {exc}",
            )

    if not cfg.project_id:
        return GEEStatus(
            library_installed=True,
            enabled=True,
            initialized=False,
            status_code="missing_project",
            auth_mode="interactive_oauth",
            project_id=None,
            export_destination=cfg.export_destination,
            message=(
                "GEE_PROJECT_ID is not set. Configure a registered Google Cloud project ID in .env "
                "after authenticating with `earthengine authenticate`."
            ),
        )

    if not _interactive_credentials_exist():
        return GEEStatus(
            library_installed=True,
            enabled=True,
            initialized=False,
            status_code="unauthenticated",
            auth_mode="interactive_oauth",
            project_id=cfg.project_id,
            export_destination=cfg.export_destination,
            message=(
                "No Earth Engine credentials found at ~/.config/earthengine/credentials. "
                "Run `earthengine authenticate` locally or configure a service account."
            ),
        )

    try:
        ee.Initialize(project=cfg.project_id)
        return GEEStatus(
            library_installed=True,
            enabled=True,
            initialized=True,
            status_code="ready",
            auth_mode="interactive_oauth",
            project_id=cfg.project_id,
            export_destination=cfg.export_destination,
            message="Initialized Google Earth Engine via interactive OAuth credentials.",
        )
    except Exception as exc:
        return GEEStatus(
            library_installed=True,
            enabled=True,
            initialized=False,
            status_code="error",
            auth_mode="interactive_oauth",
            project_id=cfg.project_id,
            export_destination=cfg.export_destination,
            message=f"Earth Engine initialization failed: {exc}",
        )


def get_gee_status(config: GEEConfig | None = None) -> GEEStatus:
    """Return current GEE configuration and authentication readiness without crashing."""
    cfg = config or GEEConfig.from_env()
    return initialize_gee(cfg, force=False)

