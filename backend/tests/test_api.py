"""Phase 12 FastAPI endpoint integration and validation tests."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

from backend.app.api.dependencies import (
    get_dataset_service,
    get_gee_registry,
    get_inference_service,
    get_model_service,
)
from backend.app.core.config import Settings, get_settings
from backend.app.main import create_app


def _png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _geotiff_bytes(arr_chw: np.ndarray, crs: str = "EPSG:32643") -> bytes:
    buf = io.BytesIO()
    c, h, w = arr_chw.shape
    with rasterio.open(
        buf,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=arr_chw.dtype,
        crs=CRS.from_user_input(crs),
        transform=from_origin(500000.0, 2100000.0, 10.0, 10.0),
    ) as dst:
        dst.write(arr_chw)
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ALLOW_PRETRAINED_DOWNLOAD", "false")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("GEE_ENABLED", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("EXPERIMENTS_DIR", str(tmp_path / "experiments"))
    monkeypatch.setenv("OUTPUTS_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_TMP_DIR", str(tmp_path / "outputs" / "tmp_uploads"))

    get_settings.cache_clear()
    get_model_service.cache_clear()
    get_inference_service.cache_clear()
    get_dataset_service.cache_clear()
    get_gee_registry.cache_clear()

    app = create_app()
    with TestClient(app) as tc:
        yield tc

    get_settings.cache_clear()
    get_model_service.cache_clear()
    get_inference_service.cache_clear()
    get_dataset_service.cache_clear()
    get_gee_registry.cache_clear()


def test_health_models_datasets_experiments_endpoints(client: TestClient) -> None:
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"
    assert r_health.json()["gee_status"]["status_code"] == "disabled"

    r_models = client.get("/api/models")
    assert r_models.status_code == 200
    model_ids = {m["model_id"] for m in r_models.json()["models"]}
    assert model_ids == {"siamese_unet", "vit", "swin"}

    r_ds = client.get("/api/datasets")
    assert r_ds.status_code == 200
    assert len(r_ds.json()["datasets"]) == 3

    r_exp = client.get("/api/experiments")
    assert r_exp.status_code == 200
    assert r_exp.json()["total_real_experiments"] == 0
    assert all(row["f1_score"] == "Not evaluated" for row in r_exp.json()["comparison_table"])


def test_upload_validation_and_error_handling(client: TestClient) -> None:
    # Unsupported format
    r_bad_ext = client.post(
        "/api/upload",
        files={
            "before_file": ("a.txt", b"hello", "text/plain"),
            "after_file": ("b.png", _png_bytes(np.zeros((32, 32, 3))), "image/png"),
        },
    )
    assert r_bad_ext.status_code == 400

    # Corrupted image bytes
    r_corrupt = client.post(
        "/api/upload",
        files={
            "before_file": ("a.png", b"corrupt-bytes", "image/png"),
            "after_file": ("b.png", _png_bytes(np.zeros((32, 32, 3))), "image/png"),
        },
    )
    assert r_corrupt.status_code == 400

    # Incompatible channels (1-channel TIFF vs 3-channel TIFF? Wait: single PNG loads 1 channel)
    gray = np.zeros((32, 32), dtype=np.uint8)
    buf_gray = io.BytesIO()
    Image.fromarray(gray, mode="L").save(buf_gray, format="PNG")
    r_ch = client.post(
        "/api/upload",
        files={
            "before_file": ("a.png", buf_gray.getvalue(), "image/png"),
            "after_file": ("b.png", _png_bytes(np.zeros((32, 32, 3))), "image/png"),
        },
    )
    assert r_ch.status_code == 400
    assert "channel" in r_ch.json()["detail"].lower()


def test_end_to_end_upload_predict_evaluate_and_reports(client: TestClient) -> None:
    arr_b = np.full((3, 64, 64), 80, dtype=np.uint8)
    arr_a = arr_b.copy()
    arr_a[:, 16:40, 16:40] = 220

    r_up = client.post(
        "/api/upload",
        files={
            "before_file": ("before.tif", _geotiff_bytes(arr_b), "image/tiff"),
            "after_file": ("after.tif", _geotiff_bytes(arr_a), "image/tiff"),
        },
    )
    assert r_up.status_code == 200
    upload_id = r_up.json()["upload_id"]
    assert r_up.json()["before_metadata"]["is_georeferenced"] is True

    r_pred = client.post(
        "/api/predict",
        json={
            "upload_id": upload_id,
            "model_name": "siamese_unet",
            "threshold": 0.5,
            "tile_size": 64,
            "stride": 64,
            "include_xai": True,
        },
    )
    assert r_pred.status_code == 200
    pred = r_pred.json()
    result_id = pred["result_id"]
    assert pred["geospatial"]["is_georeferenced"] is True
    assert pred["explainability"]["supported"] is True

    # Evaluate against ground-truth mask
    gt = np.zeros((64, 64), dtype=np.uint8)
    gt[16:40, 16:40] = 255
    gt_buf = io.BytesIO()
    Image.fromarray(gt, mode="L").save(gt_buf, format="PNG")
    r_eval = client.post(
        "/api/evaluate",
        data={"result_id": result_id},
        files={"mask_file": ("gt.png", gt_buf.getvalue(), "image/png")},
    )
    assert r_eval.status_code == 200
    assert "f1_score" in r_eval.json()["metrics"]

    # Retrieve results and export all report formats
    assert client.get("/api/results").status_code == 200
    assert client.get(f"/api/results/{result_id}").status_code == 200
    assert client.get("/api/results/..%2Fetc%2Fpasswd").status_code in {400, 404}

    for fmt, expected_ct in (
        ("json", "application/json"),
        ("csv", "text/csv"),
        ("pdf", "application/pdf"),
        ("png", "image/png"),
        ("overlay", "image/png"),
        ("geotiff", "image/tiff"),
        ("xai_panel", "image/png"),
    ):
        resp = client.get(f"/api/reports/{result_id}?format={fmt}")
        assert resp.status_code == 200
        assert expected_ct in resp.headers["content-type"]
        assert len(resp.content) > 0


def test_direct_multipart_predict_and_satellite_disabled(client: TestClient) -> None:
    b_bytes = _png_bytes(np.full((64, 64, 3), 50, dtype=np.uint8))
    a_bytes = _png_bytes(np.full((64, 64, 3), 180, dtype=np.uint8))

    r_pred = client.post(
        "/api/predict",
        data={"model_name": "vit", "threshold": "0.5", "tile_size": "64", "stride": "64"},
        files={
            "before_file": ("b.png", b_bytes, "image/png"),
            "after_file": ("a.png", a_bytes, "image/png"),
        },
    )
    assert r_pred.status_code == 200
    assert r_pred.json()["model_name"] == "vit"

    # Satellite endpoints when GEE is disabled
    assert client.get("/api/satellite/status").status_code == 200
    r_search = client.post(
        "/api/satellite/search",
        json={
            "place_name": "Bengaluru Urban Expansion Corridor",
            "before_start": "2023-01-01",
            "before_end": "2023-02-01",
            "after_start": "2024-01-01",
            "after_end": "2024-02-01",
            "max_cloud_pct": 20.0,
        },
    )
    assert r_search.status_code == 503
    assert client.get("/api/satellite/tasks/missing_task_id").status_code == 404


def test_four_band_sentinel2_geotiff_predict_and_xai(client: TestClient) -> None:
    arr_b = np.full((4, 64, 64), 1200, dtype=np.uint16)
    arr_a = arr_b.copy()
    arr_a[:, 20:44, 20:44] = 3500

    r_pred = client.post(
        "/api/predict",
        data={
            "model_name": "swin",
            "normalize_mode": "sentinel2_sr",
            "threshold": "0.5",
            "tile_size": "64",
            "stride": "64",
            "include_xai": "true",
        },
        files={
            "before_file": ("s2_before.tif", _geotiff_bytes(arr_b), "image/tiff"),
            "after_file": ("s2_after.tif", _geotiff_bytes(arr_a), "image/tiff"),
        },
    )
    assert r_pred.status_code == 200
    body = r_pred.json()
    assert body["model_name"] == "swin"
    assert body["geospatial"]["is_georeferenced"] is True
    assert body["explainability"]["supported"] is True


