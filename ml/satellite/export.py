"""Google Earth Engine GeoTIFF local download and async Drive/GCS export tracking."""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import requests

from ml.satellite.composites import build_temporal_composite
from ml.satellite.gee_client import GEEConfig
from ml.satellite.sentinel2 import AOISpec

try:
    import ee
except ImportError:  # pragma: no cover
    ee = None  # type: ignore[assignment]


class GEETaskRegistry:
    """In-memory + disk-persisted registry tracking GEE export tasks."""

    def __init__(self, state_dir: str | Path = "outputs/gee_exports") -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: dict[str, dict[str, Any]] = {}

    def register(self, task_id: str, record: dict[str, Any]) -> None:
        self._tasks[task_id] = record
        (self.state_dir / f"{task_id}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    def get(self, task_id: str) -> dict[str, Any] | None:
        if task_id in self._tasks:
            return self._tasks[task_id]
        path = self.state_dir / f"{task_id}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._tasks[task_id] = data
            return data
        return None


def _download_ee_image_as_geotiff(
    ee_image: Any,
    ee_geometry: Any,
    bands: Sequence[str],
    scale_m: int,
    target_crs: str,
    out_path: Path,
) -> Path:
    """Download an Earth Engine image directly as a single multi-band GeoTIFF."""
    url = ee_image.getDownloadURL(
        {
            "name": out_path.stem,
            "bands": list(bands),
            "region": ee_geometry,
            "scale": int(scale_m),
            "crs": target_crs,
            "format": "GEO_TIFF",
        }
    )
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = resp.content
    # GEE returns either raw GeoTIFF bytes or a zip containing a .tif
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tifs = [n for n in zf.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if not tifs:
                raise RuntimeError("GEE download archive did not contain a GeoTIFF file.")
            out_path.write_bytes(zf.read(tifs[0]))
    else:
        out_path.write_bytes(content)

    return out_path


def export_sentinel2_pair(
    aoi: AOISpec,
    before_start: str,
    before_end: str,
    after_start: str,
    after_end: str,
    bands: Sequence[str] = ("B4", "B3", "B2", "B8"),
    max_cloud_pct: float = 20.0,
    scale_m: int = 10,
    target_crs: str = "EPSG:4326",
    destination: str = "local_geotiff",
    output_dir: str | Path = "data/samples/gee_exports",
    config: GEEConfig | None = None,
    registry: GEETaskRegistry | None = None,
) -> dict[str, Any]:
    """Export aligned before-and-after Sentinel-2 composites locally or to Drive/GCS."""
    cfg = config or GEEConfig.from_env()
    comp_b = build_temporal_composite(
        aoi=aoi,
        start_date=before_start,
        end_date=before_end,
        bands=bands,
        max_cloud_pct=max_cloud_pct,
        scale_m=scale_m,
        target_crs=target_crs,
        config=cfg,
    )
    if not comp_b.get("ok"):
        return comp_b

    comp_a = build_temporal_composite(
        aoi=aoi,
        start_date=after_start,
        end_date=after_end,
        bands=bands,
        max_cloud_pct=max_cloud_pct,
        scale_m=scale_m,
        target_crs=target_crs,
        config=cfg,
    )
    if not comp_a.get("ok"):
        return comp_a

    task_id = f"gee_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    export_bands = comp_b["metadata"]["bands"]
    geom = comp_b["ee_geometry"]

    if destination == "local_geotiff":
        pair_dir = Path(output_dir) / task_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        try:
            b_tif = _download_ee_image_as_geotiff(
                comp_b["ee_image"],
                geom,
                export_bands,
                scale_m,
                target_crs,
                pair_dir / "before_s2.tif",
            )
            a_tif = _download_ee_image_as_geotiff(
                comp_a["ee_image"],
                geom,
                export_bands,
                scale_m,
                target_crs,
                pair_dir / "after_s2.tif",
            )
        except Exception as exc:
            return {
                "ok": False,
                "status_code": "local_download_failed",
                "error": (
                    f"Direct GeoTIFF download failed ({exc}). If your AOI exceeds GEE's "
                    "getDownloadURL size limit (~50 MB / 10000 px), use 'google_drive' or 'gcs' "
                    "batch export or select a smaller AOI / larger scale."
                ),
            }

        provenance = {
            "task_id": task_id,
            "status": "COMPLETED",
            "destination": "local_geotiff",
            "before_path": str(b_tif),
            "after_path": str(a_tif),
            "before_metadata": comp_b["metadata"],
            "after_metadata": comp_a["metadata"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (pair_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2), encoding="utf-8"
        )
        if registry is not None:
            registry.register(task_id, provenance)
        return {"ok": True, **provenance}

    if ee is None:
        return {"ok": False, "status_code": "missing_library", "error": "earthengine-api missing."}

    ee_task_ids: dict[str, str] = {}
    try:
        for label, img in (("before", comp_b["ee_image"]), ("after", comp_a["ee_image"])):
            desc = f"{task_id}_{label}"
            if destination == "google_drive":
                task = ee.batch.Export.image.toDrive(
                    image=img,
                    description=desc,
                    folder=cfg.drive_folder,
                    fileNamePrefix=desc,
                    region=geom,
                    scale=int(scale_m),
                    crs=target_crs,
                    fileFormat="GeoTIFF",
                    maxPixels=1e9,
                )
            elif destination == "gcs":
                if not cfg.gcs_bucket:
                    return {
                        "ok": False,
                        "status_code": "missing_gcs_bucket",
                        "error": "GEE_GCS_BUCKET must be configured for Cloud Storage exports.",
                    }
                task = ee.batch.Export.image.toCloudStorage(
                    image=img,
                    description=desc,
                    bucket=cfg.gcs_bucket,
                    fileNamePrefix=f"satellite_exports/{desc}",
                    region=geom,
                    scale=int(scale_m),
                    crs=target_crs,
                    fileFormat="GeoTIFF",
                    maxPixels=1e9,
                )
            else:
                return {
                    "ok": False,
                    "status_code": "invalid_destination",
                    "error": f"Unsupported export destination '{destination}'.",
                }
            task.start()
            ee_task_ids[label] = str(task.id)
    except Exception as exc:
        return {
            "ok": False,
            "status_code": "batch_export_error",
            "error": f"Failed to submit GEE batch export task: {exc}",
        }

    record = {
        "task_id": task_id,
        "status": "SUBMITTED",
        "destination": destination,
        "ee_task_ids": ee_task_ids,
        "before_metadata": comp_b["metadata"],
        "after_metadata": comp_a["metadata"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if registry is not None:
        registry.register(task_id, record)
    return {"ok": True, **record}


def get_export_task_status(
    task_id: str,
    registry: GEETaskRegistry,
) -> dict[str, Any]:
    """Query status of a tracked local or batch Earth Engine export task."""
    record = registry.get(task_id)
    if record is None:
        return {"found": False, "task_id": task_id, "error": f"Unknown task_id '{task_id}'."}

    if record.get("destination") == "local_geotiff":
        return {"found": True, **record}

    if ee is not None and isinstance(record.get("ee_task_ids"), dict):
        live_states = {}
        for lbl, ee_tid in record["ee_task_ids"].items():
            try:
                status_list = ee.data.getTaskStatus(ee_tid)
                live_states[lbl] = status_list[0] if status_list else {"state": "UNKNOWN"}
            except Exception as exc:
                live_states[lbl] = {"state": "ERROR", "error_message": str(exc)}
        record["live_ee_status"] = live_states
        states = {v.get("state") for v in live_states.values()}
        if states == {"COMPLETED"}:
            record["status"] = "COMPLETED"
        elif "FAILED" in states:
            record["status"] = "FAILED"
        elif "RUNNING" in states:
            record["status"] = "RUNNING"
    return {"found": True, **record}

