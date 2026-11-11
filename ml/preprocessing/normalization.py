"""Consistent radiometric normalization for RGB and multispectral satellite imagery."""

from __future__ import annotations

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _scale_to_unit_range(
    data: np.ndarray,
    mode: str,
    reflectance_scale: float = 10000.0,
) -> np.ndarray:
    """Convert raw numeric values into approximately ``[0, 1]`` reflectance/intensity."""
    arr = data.astype(np.float32, copy=True)
    if mode == "sentinel2_sr":
        return np.clip(arr / reflectance_scale, 0.0, 1.0)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    max_val = float(np.nanmax(arr[finite]))
    min_val = float(np.nanmin(arr[finite]))
    if min_val >= 0.0 and max_val <= 1.0:
        return arr
    if min_val >= 0.0 and max_val <= 255.0:
        return arr / 255.0
    if min_val >= 0.0 and max_val <= 10000.0:
        return np.clip(arr / reflectance_scale, 0.0, 1.0)
    return np.clip(arr / max(max_val, 1.0), 0.0, 1.0)


def normalize_image(
    data: np.ndarray,
    mode: str = "imagenet",
    valid_mask: np.ndarray | None = None,
    reflectance_scale: float = 10000.0,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """Normalize a ``(C, H, W)`` float32 array consistently.

    Modes:
        - ``"imagenet"``: Scales to ``[0, 1]`` then normalizes with ImageNet mean/std
          (for ``C == 3``; for ``C != 3``, uses 0.5 mean / 0.25 std per channel).
        - ``"sentinel2_sr"``: Divides Sentinel-2 L2A values by ``reflectance_scale``
          and clips to ``[0, 1]``.
        - ``"unit"``: Scales values cleanly to ``[0, 1]``.
        - ``"zscore"``: Per-channel standardization over valid pixels.
        - ``"percentile"``: Shared-mask percentile stretch to ``[0, 1]``.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected (C, H, W) array, got shape {data.shape}.")

    c = data.shape[0]
    norm_mode = (mode or "imagenet").strip().lower()

    if norm_mode == "sentinel2_sr":
        out = _scale_to_unit_range(data, "sentinel2_sr", reflectance_scale)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if norm_mode == "unit":
        out = _scale_to_unit_range(data, "unit", reflectance_scale)
        return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    if norm_mode == "imagenet":
        scaled = _scale_to_unit_range(data, "unit", reflectance_scale)
        if c == 3:
            mean = IMAGENET_MEAN[:, None, None]
            std = IMAGENET_STD[:, None, None]
        else:
            mean = np.full((c, 1, 1), 0.5, dtype=np.float32)
            std = np.full((c, 1, 1), 0.25, dtype=np.float32)
        out = (scaled - mean) / std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    mask = valid_mask if valid_mask is not None else np.all(np.isfinite(data), axis=0)
    out = np.zeros_like(data, dtype=np.float32)

    if norm_mode == "zscore":
        for ch in range(c):
            vals = data[ch][mask & np.isfinite(data[ch])]
            if vals.size == 0:
                continue
            mu = float(np.mean(vals))
            sigma = max(float(np.std(vals)), 1e-6)
            out[ch] = (data[ch] - mu) / sigma
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    if norm_mode == "percentile":
        for ch in range(c):
            vals = data[ch][mask & np.isfinite(data[ch])]
            if vals.size == 0:
                continue
            lo = float(np.percentile(vals, percentile_low))
            hi = float(np.percentile(vals, percentile_high))
            denom = max(hi - lo, 1e-6)
            out[ch] = np.clip((data[ch] - lo) / denom, 0.0, 1.0)
        return np.nan_to_num(out, nan=0.0).astype(np.float32)

    raise ValueError(
        f"Unsupported normalization mode '{mode}'. "
        "Choose from imagenet, sentinel2_sr, unit, zscore, percentile."
    )


def normalize_pair(
    before: np.ndarray,
    after: np.ndarray,
    mode: str = "imagenet",
    valid_mask: np.ndarray | None = None,
    reflectance_scale: float = 10000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical normalization parameters across a temporal image pair.

    When using data-dependent modes (``zscore`` or ``percentile``), statistics are
    computed jointly across both timestamps so genuine temporal intensity shifts
    are preserved rather than normalized away independently.
    """
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch in normalize_pair: {before.shape} vs {after.shape}")

    norm_mode = (mode or "imagenet").strip().lower()
    if norm_mode in {"imagenet", "sentinel2_sr", "unit"}:
        return (
            normalize_image(before, norm_mode, valid_mask, reflectance_scale),
            normalize_image(after, norm_mode, valid_mask, reflectance_scale),
        )

    c = before.shape[0]
    mask = valid_mask if valid_mask is not None else (
        np.all(np.isfinite(before), axis=0) & np.all(np.isfinite(after), axis=0)
    )
    out_b = np.zeros_like(before, dtype=np.float32)
    out_a = np.zeros_like(after, dtype=np.float32)

    for ch in range(c):
        vals_b = before[ch][mask & np.isfinite(before[ch])]
        vals_a = after[ch][mask & np.isfinite(after[ch])]
        joint = np.concatenate([vals_b, vals_a]) if (vals_b.size or vals_a.size) else np.array([])
        if joint.size == 0:
            continue
        if norm_mode == "zscore":
            mu = float(np.mean(joint))
            sigma = max(float(np.std(joint)), 1e-6)
            out_b[ch] = (before[ch] - mu) / sigma
            out_a[ch] = (after[ch] - mu) / sigma
        elif norm_mode == "percentile":
            lo = float(np.percentile(joint, 2.0))
            hi = float(np.percentile(joint, 98.0))
            denom = max(hi - lo, 1e-6)
            out_b[ch] = np.clip((before[ch] - lo) / denom, 0.0, 1.0)
            out_a[ch] = np.clip((after[ch] - lo) / denom, 0.0, 1.0)
        else:
            raise ValueError(f"Unsupported normalization mode '{mode}'.")

    return np.nan_to_num(out_b, nan=0.0).astype(np.float32), np.nan_to_num(
        out_a, nan=0.0
    ).astype(np.float32)

