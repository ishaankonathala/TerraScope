"""FastAPI application entry point for Satellite Land-Use Monitoring."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import datasets, health, models, prediction, reports, satellite
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description=(
            "REST API for multi-temporal satellite land-use change detection comparing "
            "Siamese U-Net, Vision Transformer (ViT), and Swin Transformer architectures."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(datasets.router)
    app.include_router(prediction.router)
    app.include_router(reports.router)
    app.include_router(satellite.router)

    return app


app = create_app()

