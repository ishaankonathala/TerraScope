"""Upload management, inference execution, artifact persistence, and evaluation."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.app.core.config import Settings
from backend.app.services.model_service import ModelService
from backend.app.utils.files import safe_resolve_child, validate_safe_id
from ml.evaluation.metrics import compute_binary_metrics
from ml.explainability import explain_prediction
from ml.geospatial import compute_geospatial_change_stats, save_georeferenced_raster
from ml.inference import ChangePredictor, PostprocessingConfig
from ml.preprocessing import load_image, load_mask, validate_image_pair


class InferenceService:
    """Orchestrates validated upload storage, model inference, XAI, and geospatial outputs."""

    def __init__(self, settings: Settings, model_service: ModelService) -> None:
        self.settings = settings
        self.model_service = model_service

    def register_upload_pair(
        self, upload_id: str, before_path: Path, after_path: Path
    ) -> dict[str, Any]:
        before = load_image(before_path, max_size_mb=self.settings.max_upload_size_mb)
        after = load_image(after_path, max_size_mb=self.settings.max_upload_size_mb)
        val = validate_image_pair(before, after)
        if not val.is_valid:
            raise ValueError("; ".join(val.errors))

        manifest = {
            "upload_id": upload_id,
            "before_path": str(before_path),
            "after_path": str(after_path),
            "before_filename": before_path.name,
            "after_filename": after_path.name,
            "before_metadata": before.metadata.to_dict(),
            "after_metadata": after.metadata.to_dict(),
            "pair_validation": {
                "is_valid": val.is_valid,
                "same_dimensions": val.same_dimensions,
                "same_channels": val.same_channels,
                "both_georeferenced": val.both_georeferenced,
                "crs_match": val.crs_match,
                "transform_match": val.transform_match,
                "warnings": val.warnings,
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        up_dir = safe_resolve_child(self.settings.upload_path, upload_id)
        (up_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def get_upload_manifest(self, upload_id: str) -> dict[str, Any]:
        uid = validate_safe_id(upload_id, "upload_id")
        manifest_path = safe_resolve_child(self.settings.upload_path, uid, "manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Upload session '{uid}' not found.")
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    def run_prediction(
        self,
        before_path: Path,
        after_path: Path,
        model_name: str = "siamese_unet",
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
        before_img = load_image(before_path, max_size_mb=self.settings.max_upload_size_mb)
        after_img = load_image(after_path, max_size_mb=self.settings.max_upload_size_mb)

        in_channels = 3
        model = self.model_service.get_or_load_model(
            model_name=model_name,
            checkpoint_name=checkpoint_name,
            in_channels=in_channels,
        )
        predictor = ChangePredictor(model=model, device=self.settings.device)
        pred = predictor.predict_images(
            before_image=before_img,
            after_image=after_img,
            threshold=threshold,
            tile_size=tile_size,
            stride=stride,
            normalize_mode=normalize_mode,
            postprocess_config=PostprocessingConfig(
                enabled=postprocess_enabled,
                morph_operation=morph_operation,
                min_region_pixels=min_region_pixels,
            ),
            return_normalized_pair=include_xai,
        )

        result_id = f"res_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        res_dir = safe_resolve_child(self.settings.outputs_path, "change_maps", result_id)
        res_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(pred.before_rgb).save(res_dir / "before.png")
        Image.fromarray(pred.after_rgb).save(res_dir / "after.png")
        Image.fromarray((pred.binary_mask > 0).astype(np.uint8) * 255, mode="L").save(
            res_dir / "binary_mask.png"
        )
        Image.fromarray(pred.overlay_rgb).save(res_dir / "overlay.png")
        np.save(res_dir / "probability_map.npy", pred.probability_map)
        np.save(res_dir / "valid_mask.npy", pred.valid_mask)

        # Save GeoTIFF change mask if input is georeferenced
        b_meta = before_img.metadata
        artifacts = {
            "before_png": f"/api/reports/{result_id}?format=before_png",
            "after_png": f"/api/reports/{result_id}?format=after_png",
            "mask_png": f"/api/reports/{result_id}?format=png",
            "overlay_png": f"/api/reports/{result_id}?format=overlay",
            "csv_report": f"/api/reports/{result_id}?format=csv",
            "pdf_report": f"/api/reports/{result_id}?format=pdf",
            "json_report": f"/api/reports/{result_id}?format=json",
        }

        if b_meta.is_georeferenced:
            save_georeferenced_raster(
                pred.binary_mask,
                b_meta,
                res_dir / "change_mask.tif",
                nodata=255,
            )
            artifacts["geotiff"] = f"/api/reports/{result_id}?format=geotiff"

        geo_stats = compute_geospatial_change_stats(pred.binary_mask, pred.valid_mask, b_meta)
        if geo_stats.get("regions_geojson"):
            (res_dir / "regions.geojson").write_text(
                geo_stats["regions_geojson"], encoding="utf-8"
            )
            artifacts["geojson"] = f"/api/reports/{result_id}?format=geojson"

        xai_summary: dict[str, Any] | None = None
        if include_xai and pred.normalized_before is not None and pred.normalized_after is not None:
            xai_dir = safe_resolve_child(self.settings.outputs_path, "attention_maps", result_id)
            xai_dir.mkdir(parents=True, exist_ok=True)
            # Resize aligned/masked normalized tensors cleanly to model native img_size
            import cv2

            sz = int(getattr(model, "img_size", 256))
            dev = next(model.parameters()).device
            bt = F.interpolate(
                torch.from_numpy(pred.normalized_before).unsqueeze(0).float().to(dev),
                size=(sz, sz),
                mode="bilinear",
                align_corners=False,
            )
            at = F.interpolate(
                torch.from_numpy(pred.normalized_after).unsqueeze(0).float().to(dev),
                size=(sz, sz),
                mode="bilinear",
                align_corners=False,
            )
            b_rgb_sz = cv2.resize(pred.before_rgb, (sz, sz))
            a_rgb_sz = cv2.resize(pred.after_rgb, (sz, sz))
            xai_out = explain_prediction(
                model=model,
                before_tensor=bt,
                after_tensor=at,
                before_rgb=b_rgb_sz,
                after_rgb=a_rgb_sz,
                output_path=xai_dir / "xai_panel.png",
            )
            if xai_out.get("supported") and "overlay_rgb" in xai_out:
                Image.fromarray(xai_out["overlay_rgb"]).save(xai_dir / "xai_overlay.png")
                artifacts["xai_panel"] = f"/api/reports/{result_id}?format=xai_panel"
                artifacts["xai_overlay"] = f"/api/reports/{result_id}?format=xai_overlay"
            xai_summary = {
                "supported": xai_out.get("supported", False),
                "method": xai_out.get("method"),
                "reason": xai_out.get("reason"),
                "disclaimer": xai_out.get("disclaimer"),
            }

        summary = {
            "result_id": result_id,
            "model_name": model_name,
            "training_status": getattr(model, "training_status", "untrained"),
            "threshold": threshold,
            "changed_pixels": pred.changed_pixels,
            "total_valid_pixels": pred.total_valid_pixels,
            "masked_invalid_pixels": pred.masked_invalid_pixels,
            "change_percentage": pred.change_percentage,
            "inference_time_sec": pred.inference_time_sec,
            "geospatial": geo_stats,
            "regions": pred.regions[:50],
            "artifacts": artifacts,
            "explainability": xai_summary,
            "metadata": pred.metadata,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (res_dir / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def get_result(self, result_id: str) -> tuple[dict[str, Any], Path]:
        rid = validate_safe_id(result_id, "result_id")
        res_dir = safe_resolve_child(self.settings.outputs_path, "change_maps", rid)
        json_path = res_dir / "result.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Result '{rid}' not found.")
        return json.loads(json_path.read_text(encoding="utf-8")), res_dir

    def list_results(self, limit: int = 30) -> list[dict[str, Any]]:
        base = self.settings.outputs_path / "change_maps"
        if not base.is_dir():
            return []
        items: list[dict[str, Any]] = []
        for res_json in sorted(base.glob("*/result.json"), reverse=True)[:limit]:
            try:
                data = json.loads(res_json.read_text(encoding="utf-8"))
                items.append(data)
            except Exception:
                continue
        return items

    def evaluate_against_mask(self, result_id: str, mask_path: Path) -> dict[str, Any]:
        result_data, res_dir = self.get_result(result_id)
        prob_path = res_dir / "probability_map.npy"
        valid_path = res_dir / "valid_mask.npy"
        if not prob_path.is_file():
            raise FileNotFoundError("Probability map missing for result evaluation.")

        probs = np.load(prob_path)
        valid = np.load(valid_path) if valid_path.is_file() else np.ones_like(probs, dtype=bool)
        gt = load_mask(mask_path, max_size_mb=self.settings.max_upload_size_mb)
        if gt.shape != probs.shape:
            import cv2

            gt = cv2.resize(gt, (probs.shape[1], probs.shape[0]), interpolation=cv2.INTER_NEAREST)

        metrics = compute_binary_metrics(
            preds=probs,
            targets=gt,
            valid_mask=valid,
            threshold=float(result_data.get("threshold", 0.5)),
            from_logits=False,
        )
        return {
            "result_id": result_id,
            "model_name": result_data.get("model_name"),
            "training_status": result_data.get("training_status"),
            "metrics": metrics.to_dict(),
            "confusion_matrix": metrics.confusion_matrix(),
        }

