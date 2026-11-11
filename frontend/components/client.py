"""Unified client bridging FastAPI endpoints and direct local backend services."""

from __future__ import annotations

import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.api.dependencies import (
    get_dataset_service,
    get_gee_config,
    get_gee_registry,
    get_inference_service,
    get_model_service,
)
from backend.app.core.config import get_settings
from ml.evaluation.benchmark import build_comparison_table, load_experiment_summaries
from ml.satellite import (
    AOISpec,
    build_temporal_composite,
    compute_gee_ndvi_preview,
    export_sentinel2_pair,
    get_export_task_status,
    get_gee_status,
    search_sentinel2_scenes,
)
from ml.utils.device import device_info_dict


class AppClient:
    """Talks to live FastAPI server when running, or falls back cleanly to local Python services."""

    def __init__(self, api_base_url: str | None = None) -> None:
        self.api_base_url = (
            api_base_url or os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
        ).rstrip("/")
        self.settings = get_settings()
        self._api_online: bool | None = None

    def is_api_online(self) -> bool:
        if self._api_online is not None:
            return self._api_online
        try:
            resp = requests.get(f"{self.api_base_url}/api/health", timeout=1.2)
            self._api_online = resp.status_code == 200
        except Exception:
            self._api_online = False
        return self._api_online

    def get_health(self) -> dict[str, Any]:
        if self.is_api_online():
            return requests.get(f"{self.api_base_url}/api/health", timeout=5).json()
        return {
            "status": "ok",
            "app_name": self.settings.app_name,
            "device": device_info_dict(self.settings.device),
            "gee_status": get_gee_status(get_gee_config(self.settings)).to_dict(),
        }

    def get_models(self) -> dict[str, Any]:
        if self.is_api_online():
            return requests.get(f"{self.api_base_url}/api/models", timeout=10).json()
        return get_model_service().get_models_catalog()

    def get_datasets(self) -> dict[str, Any]:
        if self.is_api_online():
            return requests.get(f"{self.api_base_url}/api/datasets", timeout=10).json()
        return get_dataset_service().list_datasets()

    def get_experiments(self) -> dict[str, Any]:
        if self.is_api_online():
            return requests.get(f"{self.api_base_url}/api/experiments", timeout=10).json()
        summaries = load_experiment_summaries(
            experiments_dir=self.settings.experiments_path, include_synthetic=False
        )
        comparison = build_comparison_table(
            experiments_dir=self.settings.experiments_path, include_synthetic=False
        )
        return {
            "experiments": summaries,
            "comparison_table": comparison,
            "total_real_experiments": len(summaries),
        }

    def list_results(self, limit: int = 30) -> list[dict[str, Any]]:
        if self.is_api_online():
            resp = requests.get(f"{self.api_base_url}/api/results?limit={limit}", timeout=10)
            if resp.status_code == 200:
                return resp.json().get("results", [])
        return get_inference_service().list_results(limit=limit)

    def get_result(self, result_id: str) -> dict[str, Any]:
        if self.is_api_online():
            resp = requests.get(f"{self.api_base_url}/api/results/{result_id}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        data, _ = get_inference_service().get_result(result_id)
        return data

    def predict_files(
        self,
        before_name: str,
        before_bytes: bytes,
        after_name: str,
        after_bytes: bytes,
        model_name: str,
        checkpoint_name: str | None = None,
        threshold: float = 0.5,
        tile_size: int = 256,
        stride: int = 192,
        normalize_mode: str = "imagenet",
        postprocess_enabled: bool = False,
        morph_operation: str = "none",
        min_region_pixels: int = 0,
        include_xai: bool = False,
    ) -> dict[str, Any]:
        if self.is_api_online():
            data = {
                "model_name": model_name,
                "threshold": str(threshold),
                "tile_size": str(tile_size),
                "stride": str(stride),
                "normalize_mode": normalize_mode,
                "postprocess_enabled": str(postprocess_enabled).lower(),
                "morph_operation": morph_operation,
                "min_region_pixels": str(min_region_pixels),
                "include_xai": str(include_xai).lower(),
            }
            if checkpoint_name:
                data["checkpoint_name"] = checkpoint_name
            resp = requests.post(
                f"{self.api_base_url}/api/predict",
                data=data,
                files={
                    "before_file": (before_name, before_bytes),
                    "after_file": (after_name, after_bytes),
                },
                timeout=300,
            )
            if resp.status_code != 200:
                raise ValueError(resp.json().get("detail", resp.text))
            return resp.json()

        # Local direct runtime fallback
        tmp_dir = self.settings.upload_path / f"st_{uuid.uuid4().hex[:10]}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            b_ext = Path(before_name).suffix.lower() or ".png"
            a_ext = Path(after_name).suffix.lower() or ".png"
            b_path = tmp_dir / f"before{b_ext}"
            a_path = tmp_dir / f"after{a_ext}"
            b_path.write_bytes(before_bytes)
            a_path.write_bytes(after_bytes)
            return get_inference_service().run_prediction(
                before_path=b_path,
                after_path=a_path,
                model_name=model_name,
                checkpoint_name=checkpoint_name,
                threshold=threshold,
                tile_size=tile_size,
                stride=stride,
                normalize_mode=normalize_mode,
                postprocess_enabled=postprocess_enabled,
                morph_operation=morph_operation,
                min_region_pixels=min_region_pixels,
                include_xai=include_xai,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def get_artifact_bytes(self, result_id: str, fmt: str) -> bytes | None:
        if self.is_api_online():
            resp = requests.get(
                f"{self.api_base_url}/api/reports/{result_id}?format={fmt}", timeout=30
            )
            if resp.status_code == 200:
                return resp.content
            return None

        from backend.app.services.report_service import generate_csv_report, generate_pdf_report

        data, res_dir = get_inference_service().get_result(result_id)
        if fmt == "json":
            import json

            return json.dumps(data, indent=2).encode("utf-8")
        if fmt == "csv":
            return generate_csv_report(data)
        if fmt == "pdf":
            pdf_out = self.settings.outputs_path / "reports" / f"{result_id}_report.pdf"
            generate_pdf_report(data, res_dir, pdf_out)
            return pdf_out.read_bytes()
        fmap = {
            "png": res_dir / "binary_mask.png",
            "overlay": res_dir / "overlay.png",
            "before_png": res_dir / "before.png",
            "after_png": res_dir / "after.png",
            "geotiff": res_dir / "change_mask.tif",
            "geojson": res_dir / "regions.geojson",
            "xai_panel": self.settings.outputs_path / "attention_maps" / result_id / "xai_panel.png",
            "xai_overlay": self.settings.outputs_path / "attention_maps" / result_id / "xai_overlay.png",
        }
        target = fmap.get(fmt)
        if target and target.is_file():
            return target.read_bytes()
        return None

    def get_satellite_status(self) -> dict[str, Any]:
        if self.is_api_online():
            return requests.get(f"{self.api_base_url}/api/satellite/status", timeout=10).json()
        from ml.satellite import PRESET_REGIONS

        st_dict = get_gee_status(get_gee_config(self.settings)).to_dict()
        st_dict["preset_regions"] = PRESET_REGIONS
        return st_dict

    def search_satellite(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.is_api_online():
            resp = requests.post(
                f"{self.api_base_url}/api/satellite/search", json=payload, timeout=60
            )
            if resp.status_code != 200:
                raise RuntimeError(resp.json().get("detail", resp.text))
            return resp.json()
        cfg = get_gee_config(self.settings)
        aoi = AOISpec(
            bbox=payload.get("bbox"),
            place_name=payload.get("place_name"),
            geojson_geometry=payload.get("geojson_geometry"),
        )
        return {
            "before": search_sentinel2_scenes(
                aoi, payload["before_start"], payload["before_end"], payload["max_cloud_pct"], config=cfg
            ),
            "after": search_sentinel2_scenes(
                aoi, payload["after_start"], payload["after_end"], payload["max_cloud_pct"], config=cfg
            ),
        }

    def preview_satellite(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.is_api_online():
            resp = requests.post(
                f"{self.api_base_url}/api/satellite/preview", json=payload, timeout=120
            )
            if resp.status_code != 200:
                raise RuntimeError(resp.json().get("detail", resp.text))
            return resp.json()
        cfg = get_gee_config(self.settings)
        aoi = AOISpec(
            bbox=payload.get("bbox"),
            place_name=payload.get("place_name"),
            geojson_geometry=payload.get("geojson_geometry"),
        )
        cb = build_temporal_composite(
            aoi,
            payload["before_start"],
            payload["before_end"],
            bands=payload["bands"],
            max_cloud_pct=payload["max_cloud_pct"],
            scale_m=payload["scale_m"],
            target_crs=payload.get("target_crs", "EPSG:4326"),
            apply_scl_mask=payload.get("apply_scl_mask", True),
            config=cfg,
        )
        ca = build_temporal_composite(
            aoi,
            payload["after_start"],
            payload["after_end"],
            bands=payload["bands"],
            max_cloud_pct=payload["max_cloud_pct"],
            scale_m=payload["scale_m"],
            target_crs=payload.get("target_crs", "EPSG:4326"),
            apply_scl_mask=payload.get("apply_scl_mask", True),
            config=cfg,
        )
        if not cb.get("ok"):
            raise RuntimeError(cb.get("error", "Before composite failed."))
        if not ca.get("ok"):
            raise RuntimeError(ca.get("error", "After composite failed."))
        ndvi = None
        if "B4" in payload["bands"] and "B8" in payload["bands"]:
            ndvi = compute_gee_ndvi_preview(
                cb["ee_image"], ca["ee_image"], cb["ee_geometry"], scale_m=max(payload["scale_m"], 20)
            )
        return {
            "before": {"thumb_url": cb["thumb_url"], "metadata": cb["metadata"]},
            "after": {"thumb_url": ca["thumb_url"], "metadata": ca["metadata"]},
            "ndvi": ndvi,
        }

    def export_satellite(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.is_api_online():
            resp = requests.post(
                f"{self.api_base_url}/api/satellite/export", json=payload, timeout=180
            )
            if resp.status_code != 200:
                raise RuntimeError(resp.json().get("detail", resp.text))
            return resp.json()
        cfg = get_gee_config(self.settings)
        aoi = AOISpec(
            bbox=payload.get("bbox"),
            place_name=payload.get("place_name"),
            geojson_geometry=payload.get("geojson_geometry"),
        )
        res = export_sentinel2_pair(
            aoi=aoi,
            before_start=payload["before_start"],
            before_end=payload["before_end"],
            after_start=payload["after_start"],
            after_end=payload["after_end"],
            bands=payload["bands"],
            max_cloud_pct=payload["max_cloud_pct"],
            scale_m=payload["scale_m"],
            target_crs=payload.get("target_crs", "EPSG:4326"),
            destination=payload.get("destination", "local_geotiff"),
            output_dir=self.settings.data_path / "samples" / "gee_exports",
            config=cfg,
            registry=get_gee_registry(),
        )
        if not res.get("ok"):
            raise RuntimeError(res.get("error", "Export failed."))
        return res

    def get_satellite_task(self, task_id: str) -> dict[str, Any]:
        if self.is_api_online():
            resp = requests.get(
                f"{self.api_base_url}/api/satellite/tasks/{task_id}", timeout=20
            )
            resp.raise_for_status()
            return resp.json()
        return get_export_task_status(task_id, get_gee_registry())


def get_app_client() -> AppClient:
    return AppClient()

