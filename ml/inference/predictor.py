"""Reusable change-detection inference engine with tiled overlap blending."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.inference.postprocessing import PostprocessingConfig, postprocess_mask
from ml.models import BaseChangeDetector, get_model
from ml.preprocessing import (
    CloudMaskConfig,
    LoadedImage,
    align_image_pair,
    apply_cloud_mask,
    create_blend_window,
    extract_patches,
    load_image,
    normalize_pair,
    stitch_tiles,
)
from ml.training.callbacks import load_checkpoint
from ml.utils.device import detect_device


@dataclass
class PredictionResult:
    """Structured output produced by the change-detection inference pipeline."""

    probability_map: np.ndarray  # (H, W) float32 in [0, 1]
    binary_mask: np.ndarray  # (H, W) uint8 {0, 1}
    raw_binary_mask: np.ndarray  # (H, W) uint8 before optional post-processing
    valid_mask: np.ndarray  # (H, W) bool
    before_rgb: np.ndarray  # (H, W, 3) uint8
    after_rgb: np.ndarray  # (H, W, 3) uint8
    overlay_rgb: np.ndarray  # (H, W, 3) uint8
    changed_pixels: int
    total_valid_pixels: int
    masked_invalid_pixels: int
    change_percentage: float
    threshold: float
    inference_time_sec: float
    regions: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    normalized_before: np.ndarray | None = None
    normalized_after: np.ndarray | None = None


def create_change_overlay(
    after_rgb: np.ndarray,
    binary_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    alpha: float = 0.48,
) -> np.ndarray:
    """Overlay predicted change regions (red-orange) and invalid mask (cyan tint) on after RGB."""
    base = after_rgb.astype(np.float32).copy()
    changed = binary_mask > 0
    if valid_mask is not None:
        changed = changed & valid_mask

    highlight = np.array([255.0, 64.0, 48.0], dtype=np.float32)
    base[changed] = (1.0 - alpha) * base[changed] + alpha * highlight

    if valid_mask is not None:
        invalid = ~valid_mask
        if np.any(invalid):
            invalid_tint = np.array([70.0, 160.0, 195.0], dtype=np.float32)
            base[invalid] = 0.55 * base[invalid] + 0.45 * invalid_tint

    return np.clip(np.round(base), 0, 255).astype(np.uint8)


class ChangePredictor:
    """Runs end-to-end preprocessing, overlapping tiled inference, and post-processing."""

    def __init__(
        self,
        model: BaseChangeDetector | None = None,
        model_name: str = "siamese_unet",
        checkpoint_path: str | Path | None = None,
        in_channels: int = 3,
        img_size: int = 256,
        pretrained: bool = True,
        allow_random_fallback: bool = True,
        device: str = "auto",
        backbone_name: str | None = None,
    ) -> None:
        self.device = detect_device(device)
        if model is None:
            self.model = get_model(
                model_name,
                in_channels=in_channels,
                num_classes=1,
                img_size=img_size,
                pretrained=pretrained,
                allow_random_fallback=allow_random_fallback,
                backbone_name=backbone_name,
            )
        else:
            self.model = model

        if checkpoint_path is not None:
            load_checkpoint(checkpoint_path, self.model, map_location=self.device)

        self.model.to(self.device)
        self.model.eval()
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None

    @staticmethod
    def _strip_quality_bands_if_needed(
        image: LoadedImage, expected_channels: int
    ) -> tuple[LoadedImage, np.ndarray | None, np.ndarray | None]:
        """Separate auxiliary SCL/cloud probability bands if present before model input."""
        if not image.metadata.band_names:
            return image, None, None

        names = [n.strip().upper() for n in image.metadata.band_names]
        scl_band = None
        prob_band = None
        keep_indices: list[int] = []
        keep_names: list[str] = []

        for idx, name in enumerate(names):
            if name == "SCL":
                scl_band = image.data[idx]
            elif name in {"MSK_CLDPRB", "CLOUD_PROBABILITY"}:
                prob_band = image.data[idx]
            else:
                keep_indices.append(idx)
                keep_names.append(image.metadata.band_names[idx])

        if len(keep_indices) == image.channels or not keep_indices:
            return image, scl_band, prob_band

        # Reduce to spectral bands matching expected_channels if quality bands were stripped
        if len(keep_indices) > expected_channels:
            keep_indices = keep_indices[:expected_channels]
            keep_names = keep_names[:expected_channels]

        new_meta = image.metadata
        new_meta.count = len(keep_indices)
        new_meta.band_names = keep_names
        trimmed = LoadedImage(
            data=image.data[keep_indices],
            valid_mask=image.valid_mask,
            metadata=new_meta,
            raw_dtype=image.raw_dtype,
            has_spectral_quality_bands=image.has_spectral_quality_bands,
            warnings=list(image.warnings),
        )
        return trimmed, scl_band, prob_band

    def prepare_inputs(
        self,
        before_image: LoadedImage,
        after_image: LoadedImage,
        cloud_config: CloudMaskConfig | None = None,
        use_ecc: bool = False,
        rgb_indices: tuple[int, int, int] = (0, 1, 2),
    ) -> tuple[LoadedImage, LoadedImage, Any, Any, Any]:
        """Strip quality bands, mask clouds/shadows, match channel depth, and spatially align pair."""
        expected_ch = int(getattr(self.model, "in_channels", 3))

        b_clean, b_scl, b_prob = self._strip_quality_bands_if_needed(before_image, expected_ch)
        a_clean, a_scl, a_prob = self._strip_quality_bands_if_needed(after_image, expected_ch)

        if b_scl is not None or b_prob is not None:
            b_clean, b_cloud_rep = apply_cloud_mask(
                b_clean, scl_band=b_scl, cloud_prob_band=b_prob, config=cloud_config
            )
        else:
            b_clean, b_cloud_rep = apply_cloud_mask(b_clean, config=cloud_config)

        if a_scl is not None or a_prob is not None:
            a_clean, a_cloud_rep = apply_cloud_mask(
                a_clean, scl_band=a_scl, cloud_prob_band=a_prob, config=cloud_config
            )
        else:
            a_clean, a_cloud_rep = apply_cloud_mask(a_clean, config=cloud_config)

        # If model expects 3 RGB channels but GeoTIFF contains >3 spectral bands, select RGB subset
        if b_clean.channels > expected_ch and expected_ch == 3:
            idx_list = list(rgb_indices)
            b_clean = LoadedImage(
                data=b_clean.data[idx_list],
                valid_mask=b_clean.valid_mask,
                metadata=b_clean.metadata,
                raw_dtype=b_clean.raw_dtype,
                has_spectral_quality_bands=b_clean.has_spectral_quality_bands,
                warnings=list(b_clean.warnings)
                + [f"Selected bands {idx_list} for 3-channel model inference."],
            )
            a_clean = LoadedImage(
                data=a_clean.data[idx_list],
                valid_mask=a_clean.valid_mask,
                metadata=a_clean.metadata,
                raw_dtype=a_clean.raw_dtype,
                has_spectral_quality_bands=a_clean.has_spectral_quality_bands,
                warnings=list(a_clean.warnings),
            )

        b_al, a_al, align_rep = align_image_pair(b_clean, a_clean, use_ecc=use_ecc)
        return b_al, a_al, align_rep, b_cloud_rep, a_cloud_rep

    def predict_images(
        self,
        before_image: LoadedImage,
        after_image: LoadedImage,
        threshold: float = 0.5,
        tile_size: int = 256,
        stride: int = 192,
        normalize_mode: str = "imagenet",
        cloud_config: CloudMaskConfig | None = None,
        postprocess_config: PostprocessingConfig | None = None,
        use_ecc: bool = False,
        rgb_indices: tuple[int, int, int] = (0, 1, 2),
        return_normalized_pair: bool = False,
    ) -> PredictionResult:
        """Run change-detection inference on aligned/loaded image objects."""
        t0 = time.perf_counter()
        b_al, a_al, align_rep, b_cloud_rep, a_cloud_rep = self.prepare_inputs(
            before_image=before_image,
            after_image=after_image,
            cloud_config=cloud_config,
            use_ecc=use_ecc,
            rgb_indices=rgb_indices,
        )
        valid_mask = b_al.valid_mask & a_al.valid_mask
        h, w = b_al.height, b_al.width

        before_rgb = b_al.to_rgb_uint8()
        after_rgb = a_al.to_rgb_uint8()

        b_norm, a_norm = normalize_pair(b_al.data, a_al.data, mode=normalize_mode, valid_mask=valid_mask)

        # Tiled overlapping inference with smooth Hann-window blending
        b_tiles, specs, _ = extract_patches(b_norm, tile_size=tile_size, stride=stride, pad_if_smaller=True)
        a_tiles, _, _ = extract_patches(a_norm, tile_size=tile_size, stride=stride, pad_if_smaller=True)

        prob_tiles: list[np.ndarray] = []
        batch_size = 4
        with torch.no_grad():
            for i in range(0, len(b_tiles), batch_size):
                bt = torch.from_numpy(np.stack(b_tiles[i : i + batch_size])).float().to(self.device)
                at = torch.from_numpy(np.stack(a_tiles[i : i + batch_size])).float().to(self.device)
                probs = self.model.predict_proba(bt, at)
                if probs.shape[1] == 1:
                    probs_2d = probs[:, 0]
                else:
                    # Multi-class future interface: total non-background change probability
                    probs_2d = 1.0 - probs[:, 0]
                probs_np = probs_2d.detach().cpu().numpy().astype(np.float32)
                for j in range(probs_np.shape[0]):
                    prob_tiles.append(probs_np[j])

        window = create_blend_window(tile_size, mode="hann")
        prob_map = stitch_tiles(prob_tiles, specs, out_height=h, out_width=w, window=window)
        prob_map = np.clip(prob_map, 0.0, 1.0)
        # Do not predict changes on masked invalid/cloud pixels
        prob_map = np.where(valid_mask, prob_map, 0.0).astype(np.float32)

        raw_binary = ((prob_map >= threshold) & valid_mask).astype(np.uint8)
        final_binary, post_info = postprocess_mask(
            raw_binary, valid_mask=valid_mask, config=postprocess_config
        )

        total_pixels = int(h * w)
        total_valid = int(np.sum(valid_mask))
        changed_px = int(np.sum(final_binary > 0))
        change_pct = (changed_px / total_valid * 100.0) if total_valid > 0 else 0.0

        overlay_rgb = create_change_overlay(after_rgb, final_binary, valid_mask=valid_mask)
        elapsed = round(time.perf_counter() - t0, 4)

        return PredictionResult(
            probability_map=prob_map,
            binary_mask=final_binary,
            raw_binary_mask=raw_binary,
            valid_mask=valid_mask,
            before_rgb=before_rgb,
            after_rgb=after_rgb,
            overlay_rgb=overlay_rgb,
            changed_pixels=changed_px,
            total_valid_pixels=total_valid,
            masked_invalid_pixels=total_pixels - total_valid,
            change_percentage=round(change_pct, 4),
            threshold=threshold,
            inference_time_sec=elapsed,
            regions=post_info["regions"],
            metadata={
                "model_info": self.model.model_info() if hasattr(self.model, "model_info") else {},
                "checkpoint_path": self.checkpoint_path,
                "device": str(self.device),
                "before_metadata": b_al.metadata.to_dict(),
                "after_metadata": a_al.metadata.to_dict(),
                "alignment": align_rep.to_dict(),
                "cloud_mask_before": b_cloud_rep.to_dict(),
                "cloud_mask_after": a_cloud_rep.to_dict(),
                "postprocessing": post_info["summary"],
                "tile_size": tile_size,
                "stride": stride,
            },
            normalized_before=b_norm if return_normalized_pair else None,
            normalized_after=a_norm if return_normalized_pair else None,
        )

    def predict_files(
        self,
        before_path: str | Path,
        after_path: str | Path,
        max_size_mb: float = 50.0,
        **kwargs: Any,
    ) -> PredictionResult:
        """Load two image files from disk and run full change-detection inference."""
        before = load_image(before_path, max_size_mb=max_size_mb)
        after = load_image(after_path, max_size_mb=max_size_mb)
        return self.predict_images(before, after, **kwargs)

