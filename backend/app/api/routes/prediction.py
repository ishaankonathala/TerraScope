"""Upload validation and change-detection prediction endpoints."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from backend.app.api.dependencies import get_inference_service
from backend.app.core.config import get_settings
from backend.app.schemas.api_schemas import (
    PredictByUploadRequest,
    PredictionResponse,
    UploadPairResponse,
)
from backend.app.services.inference_service import InferenceService
from backend.app.utils.files import safe_resolve_child, save_upload_validated
from ml.preprocessing.image_loader import ImageValidationError

router = APIRouter(tags=["Prediction"])


@router.post("/api/upload", response_model=UploadPairResponse)
async def upload_image_pair(
    before_file: UploadFile = File(...),
    after_file: UploadFile = File(...),
    inference_service: InferenceService = Depends(get_inference_service),
) -> UploadPairResponse:
    """Upload and validate a before/after satellite image pair."""
    settings = get_settings()
    upload_id = f"up_{uuid.uuid4().hex[:12]}"
    up_dir = safe_resolve_child(settings.upload_path, upload_id)

    try:
        b_path = await save_upload_validated(
            before_file, up_dir, "before", max_size_mb=settings.max_upload_size_mb
        )
        a_path = await save_upload_validated(
            after_file, up_dir, "after", max_size_mb=settings.max_upload_size_mb
        )
        manifest = inference_service.register_upload_pair(upload_id, b_path, a_path)
    except HTTPException:
        shutil.rmtree(up_dir, ignore_errors=True)
        raise
    except (ImageValidationError, ValueError) as exc:
        shutil.rmtree(up_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return UploadPairResponse(
        upload_id=manifest["upload_id"],
        before_filename=manifest["before_filename"],
        after_filename=manifest["after_filename"],
        before_metadata=manifest["before_metadata"],
        after_metadata=manifest["after_metadata"],
        pair_validation=manifest["pair_validation"],
    )


@router.post("/api/predict", response_model=PredictionResponse)
async def predict_change(
    request: Request,
    inference_service: InferenceService = Depends(get_inference_service),
) -> PredictionResponse:
    """Run change detection from a previously uploaded ``upload_id`` or direct multipart files."""
    settings = get_settings()
    content_type = (request.headers.get("content-type") or "").lower()

    temp_dir: Path | None = None
    try:
        if "application/json" in content_type:
            raw = await request.json()
            req = PredictByUploadRequest.model_validate(raw)
            manifest = inference_service.get_upload_manifest(req.upload_id)
            before_path = Path(manifest["before_path"])
            after_path = Path(manifest["after_path"])
        elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            before_upload = form.get("before_file")
            after_upload = form.get("after_file")
            upload_id_field = form.get("upload_id")

            payload: dict[str, Any] = {
                "upload_id": str(upload_id_field) if upload_id_field else "direct_upload",
                "model_name": str(form.get("model_name") or "siamese_unet"),
                "checkpoint_name": str(form["checkpoint_name"]) if form.get("checkpoint_name") else None,
                "threshold": float(form.get("threshold") or 0.5),
                "tile_size": int(form.get("tile_size") or 256),
                "stride": int(form.get("stride") or 192),
                "normalize_mode": str(form.get("normalize_mode") or "imagenet"),
                "postprocess_enabled": str(form.get("postprocess_enabled") or "false").lower() == "true",
                "morph_operation": str(form.get("morph_operation") or "none"),
                "min_region_pixels": int(form.get("min_region_pixels") or 0),
                "include_xai": str(form.get("include_xai") or "false").lower() == "true",
            }
            req = PredictByUploadRequest.model_validate(payload)

            if hasattr(before_upload, "read") and hasattr(after_upload, "read"):
                temp_id = f"tmp_{uuid.uuid4().hex[:12]}"
                temp_dir = safe_resolve_child(settings.upload_path, temp_id)
                before_path = await save_upload_validated(
                    before_upload, temp_dir, "before", max_size_mb=settings.max_upload_size_mb
                )
                after_path = await save_upload_validated(
                    after_upload, temp_dir, "after", max_size_mb=settings.max_upload_size_mb
                )
            elif upload_id_field:
                manifest = inference_service.get_upload_manifest(str(upload_id_field))
                before_path = Path(manifest["before_path"])
                after_path = Path(manifest["after_path"])
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Provide either 'upload_id' or both 'before_file' and 'after_file'.",
                )
        else:
            raise HTTPException(
                status_code=415,
                detail="Content-Type must be application/json or multipart/form-data.",
            )

        result = inference_service.run_prediction(
            before_path=before_path,
            after_path=after_path,
            model_name=req.model_name,
            checkpoint_name=req.checkpoint_name,
            threshold=req.threshold,
            tile_size=req.tile_size,
            stride=req.stride,
            normalize_mode=req.normalize_mode,
            postprocess_enabled=req.postprocess_enabled,
            morph_operation=req.morph_operation,
            min_region_pixels=req.min_region_pixels,
            include_xai=req.include_xai,
        )
        return PredictionResponse(**result)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ImageValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)

