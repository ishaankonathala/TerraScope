"""Evaluation, experiment logs, result retrieval, and report export endpoints."""

from __future__ import annotations

import shutil
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from backend.app.api.dependencies import get_inference_service
from backend.app.core.config import get_settings
from backend.app.services.inference_service import InferenceService
from backend.app.services.report_service import generate_csv_report, generate_pdf_report
from backend.app.utils.files import safe_resolve_child, save_upload_validated
from ml.evaluation.benchmark import build_comparison_table, load_experiment_summaries
from ml.preprocessing.image_loader import ImageValidationError

router = APIRouter(tags=["Evaluation & Reports"])


@router.post("/api/evaluate")
async def evaluate_prediction(
    result_id: str = Form(...),
    mask_file: UploadFile = File(...),
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict[str, Any]:
    """Evaluate a saved prediction against an uploaded ground-truth binary mask."""
    settings = get_settings()
    eval_tmp = safe_resolve_child(settings.upload_path, f"eval_{uuid.uuid4().hex[:10]}")
    try:
        mask_path = await save_upload_validated(
            mask_file, eval_tmp, "mask", max_size_mb=settings.max_upload_size_mb
        )
        return inference_service.evaluate_against_mask(result_id=result_id, mask_path=mask_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ImageValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(eval_tmp, ignore_errors=True)


@router.get("/api/experiments")
def list_experiments() -> dict[str, Any]:
    """Return real logged experiments and 3-model comparison table (no synthetic data)."""
    settings = get_settings()
    summaries = load_experiment_summaries(
        experiments_dir=settings.experiments_path, include_synthetic=False
    )
    comparison = build_comparison_table(
        experiments_dir=settings.experiments_path, include_synthetic=False
    )
    return {
        "experiments": summaries,
        "comparison_table": comparison,
        "total_real_experiments": len(summaries),
    }


@router.get("/api/results")
def list_results(
    limit: int = Query(default=20, ge=1, le=100),
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict[str, Any]:
    """List recent saved change-detection results."""
    items = inference_service.list_results(limit=limit)
    return {"results": items, "count": len(items)}


@router.get("/api/results/{result_id}")
def get_result_detail(
    result_id: str,
    inference_service: InferenceService = Depends(get_inference_service),
) -> dict[str, Any]:
    """Retrieve stored prediction metadata and metrics by ``result_id``."""
    try:
        data, _ = inference_service.get_result(result_id)
        return data
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/reports/{result_id}")
def download_report(
    result_id: str,
    format: str = Query(
        default="json",
        pattern="^(json|csv|pdf|png|overlay|before_png|after_png|geotiff|geojson|xai_panel|xai_overlay)$",
    ),
    inference_service: InferenceService = Depends(get_inference_service),
) -> Response:
    """Export metrics, binary change mask, GeoTIFF, GeoJSON, overlay, or PDF report."""
    settings = get_settings()
    try:
        data, res_dir = inference_service.get_result(result_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if format == "json":
        return JSONResponse(content=data)

    if format == "csv":
        csv_bytes = generate_csv_report(data)
        return Response(
            content=csv_bytes,
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{result_id}_metrics.csv"'},
        )

    if format == "pdf":
        pdf_path = safe_resolve_child(settings.outputs_path, "reports", f"{result_id}_report.pdf")
        generate_pdf_report(data, res_dir, pdf_path)
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=f"{result_id}_report.pdf",
        )

    file_map = {
        "png": res_dir / "binary_mask.png",
        "overlay": res_dir / "overlay.png",
        "before_png": res_dir / "before.png",
        "after_png": res_dir / "after.png",
        "geotiff": res_dir / "change_mask.tif",
        "geojson": res_dir / "regions.geojson",
        "xai_panel": safe_resolve_child(settings.outputs_path, "attention_maps", result_id, "xai_panel.png"),
        "xai_overlay": safe_resolve_child(settings.outputs_path, "attention_maps", result_id, "xai_overlay.png"),
    }
    target = file_map.get(format)
    if target is None or not target.is_file():
        raise HTTPException(
            status_code=404, detail=f"Artifact format '{format}' is unavailable for result '{result_id}'."
        )

    media_types = {
        "png": "image/png",
        "overlay": "image/png",
        "before_png": "image/png",
        "after_png": "image/png",
        "xai_panel": "image/png",
        "xai_overlay": "image/png",
        "geotiff": "image/tiff",
        "geojson": "application/geo+json",
    }
    return FileResponse(path=str(target), media_type=media_types[format], filename=target.name)

