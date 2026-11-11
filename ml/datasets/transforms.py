"""Paired temporal geometric and radiometric augmentations for change detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch

from ml.preprocessing.normalization import normalize_pair


@dataclass
class PairedTransformConfig:
    """Configuration for paired temporal change-detection transformations."""

    output_size: tuple[int, int] | None = (256, 256)
    random_crop: bool = False
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rot90_prob: float = 0.5
    rotation_prob: float = 0.0
    max_rotation_deg: float = 15.0
    brightness_prob: float = 0.3
    brightness_delta: float = 0.12
    contrast_prob: float = 0.3
    contrast_range: tuple[float, float] = (0.88, 1.12)
    gaussian_noise_prob: float = 0.2
    gaussian_noise_std: float = 0.02
    normalize_mode: str = "imagenet"
    reflectance_scale: float = 10000.0


class PairedChangeTransform:
    """Applies identical geometric transformations to temporal image pairs and masks.

    Input arrays are expected in ``(C, H, W)`` float32 for images and ``(H, W)``
    for optional binary change masks and valid-pixel masks. Radiometric adjustments
    avoid unrealistic hue/saturation distortions for satellite imagery.
    """

    def __init__(
        self,
        config: PairedTransformConfig | None = None,
        is_train: bool = True,
        seed: int | None = None,
    ) -> None:
        self.config = config or PairedTransformConfig()
        self.is_train = is_train
        self.rng = np.random.default_rng(seed)

    def _random_crop_or_resize(
        self,
        before: np.ndarray,
        after: np.ndarray,
        mask: np.ndarray | None,
        valid_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        target = self.config.output_size
        if target is None:
            return before, after, mask, valid_mask

        th, tw = target
        _, h, w = before.shape

        if self.is_train and self.config.random_crop and h >= th and w >= tw:
            y0 = 0 if h == th else int(self.rng.integers(0, h - th + 1))
            x0 = 0 if w == tw else int(self.rng.integers(0, w - tw + 1))
            before = before[:, y0 : y0 + th, x0 : x0 + tw]
            after = after[:, y0 : y0 + th, x0 : x0 + tw]
            if mask is not None:
                mask = mask[y0 : y0 + th, x0 : x0 + tw]
            if valid_mask is not None:
                valid_mask = valid_mask[y0 : y0 + th, x0 : x0 + tw]
            return before, after, mask, valid_mask

        if (h, w) == (th, tw):
            return before, after, mask, valid_mask

        def _resize_chw(arr: np.ndarray) -> np.ndarray:
            hwc = np.transpose(arr, (1, 2, 0))
            res = cv2.resize(hwc, (tw, th), interpolation=cv2.INTER_LINEAR)
            if res.ndim == 2:
                res = res[:, :, None]
            return np.transpose(res, (2, 0, 1)).astype(np.float32)

        before = _resize_chw(before)
        after = _resize_chw(after)
        if mask is not None:
            mask = cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
        if valid_mask is not None:
            valid_mask = (
                cv2.resize(valid_mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)
                > 0
            )
        return before, after, mask, valid_mask

    def _apply_geometric(
        self,
        before: np.ndarray,
        after: np.ndarray,
        mask: np.ndarray | None,
        valid_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        # 1. Horizontal flip
        if self.rng.random() < self.config.hflip_prob:
            before = before[:, :, ::-1].copy()
            after = after[:, :, ::-1].copy()
            if mask is not None:
                mask = mask[:, ::-1].copy()
            if valid_mask is not None:
                valid_mask = valid_mask[:, ::-1].copy()

        # 2. Vertical flip
        if self.rng.random() < self.config.vflip_prob:
            before = before[:, ::-1, :].copy()
            after = after[:, ::-1, :].copy()
            if mask is not None:
                mask = mask[::-1, :].copy()
            if valid_mask is not None:
                valid_mask = valid_mask[::-1, :].copy()

        # 3. Exact 90-degree rotation (only when square or preserving size)
        _, h, w = before.shape
        if h == w and self.rng.random() < self.config.rot90_prob:
            k = int(self.rng.integers(1, 4))
            before = np.rot90(before, k=k, axes=(1, 2)).copy()
            after = np.rot90(after, k=k, axes=(1, 2)).copy()
            if mask is not None:
                mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
            if valid_mask is not None:
                valid_mask = np.rot90(valid_mask, k=k, axes=(0, 1)).copy()

        # 4. Optional small-angle continuous rotation
        if self.config.rotation_prob > 0 and self.rng.random() < self.config.rotation_prob:
            angle = float(self.rng.uniform(-self.config.max_rotation_deg, self.config.max_rotation_deg))
            _, cur_h, cur_w = before.shape
            center = (cur_w / 2.0, cur_h / 2.0)
            mat = cv2.getRotationMatrix2D(center, angle, 1.0)

            def _rot_chw(arr: np.ndarray) -> np.ndarray:
                hwc = np.transpose(arr, (1, 2, 0))
                rot = cv2.warpAffine(
                    hwc,
                    mat,
                    (cur_w, cur_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                if rot.ndim == 2:
                    rot = rot[:, :, None]
                return np.transpose(rot, (2, 0, 1)).astype(np.float32)

            before = _rot_chw(before)
            after = _rot_chw(after)
            if mask is not None:
                mask = cv2.warpAffine(
                    mask.astype(np.uint8),
                    mat,
                    (cur_w, cur_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
            if valid_mask is not None:
                valid_mask = (
                    cv2.warpAffine(
                        valid_mask.astype(np.uint8),
                        mat,
                        (cur_w, cur_h),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    > 0
                )

        return before, after, mask, valid_mask

    def _apply_radiometric(
        self,
        before: np.ndarray,
        after: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply mild realistic intensity scaling & sensor noise in [0, 1]-like space."""
        # Scale temporarily to [0, 1] only if uint8-like, otherwise operate proportionally
        def _jitter(arr: np.ndarray) -> np.ndarray:
            out = arr.astype(np.float32, copy=True)
            finite = np.isfinite(out)
            scale = float(np.nanmax(out[finite])) if np.any(finite) else 1.0
            scale = max(scale, 1.0)
            if self.rng.random() < self.config.contrast_prob:
                alpha = float(
                    self.rng.uniform(self.config.contrast_range[0], self.config.contrast_range[1])
                )
                mean_val = float(np.nanmean(out[finite])) if np.any(finite) else 0.0
                out = (out - mean_val) * alpha + mean_val
            if self.rng.random() < self.config.brightness_prob:
                delta = float(
                    self.rng.uniform(-self.config.brightness_delta, self.config.brightness_delta)
                ) * scale
                out = out + delta
            if self.rng.random() < self.config.gaussian_noise_prob:
                noise = self.rng.normal(
                    0.0, self.config.gaussian_noise_std * scale, size=out.shape
                ).astype(np.float32)
                out = out + noise
            return np.clip(out, 0.0, max(scale * 1.25, 1.0)).astype(np.float32)

        return _jitter(before), _jitter(after)

    def __call__(
        self,
        before: np.ndarray,
        after: np.ndarray,
        mask: np.ndarray | None = None,
        valid_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Transform ``(C, H, W)`` arrays and optional masks into PyTorch tensors."""
        b = before.astype(np.float32, copy=True)
        a = after.astype(np.float32, copy=True)
        m = (mask > 0).astype(np.uint8) if mask is not None else None
        vm = valid_mask.astype(bool, copy=True) if valid_mask is not None else np.ones(b.shape[1:], dtype=bool)

        b, a, m, vm = self._random_crop_or_resize(b, a, m, vm)

        if self.is_train:
            b, a, m, vm = self._apply_geometric(b, a, m, vm)
            b, a = self._apply_radiometric(b, a)

        b_norm, a_norm = normalize_pair(
            b,
            a,
            mode=self.config.normalize_mode,
            valid_mask=vm,
            reflectance_scale=self.config.reflectance_scale,
        )

        result: dict[str, Any] = {
            "before": torch.from_numpy(np.ascontiguousarray(b_norm)).float(),
            "after": torch.from_numpy(np.ascontiguousarray(a_norm)).float(),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(vm)).bool(),
        }
        if m is not None:
            result["mask"] = torch.from_numpy(np.ascontiguousarray((m > 0).astype(np.float32))).float()
        return result

