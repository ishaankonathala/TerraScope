"""Patch tiling and smooth overlap blending for large satellite imagery."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """Spatial coordinates of an extracted tile within a larger raster."""

    row: int
    col: int
    y0: int
    y1: int
    x0: int
    x1: int


def compute_tile_grid(
    height: int,
    width: int,
    tile_size: int = 256,
    stride: int | None = None,
) -> list[TileSpec]:
    """Compute a complete, edge-covered grid of ``tile_size x tile_size`` patches.

    If ``height`` or ``width`` is smaller than ``tile_size``, returns a single
    tile bounded by the image extent (caller pads up to ``tile_size``). Otherwise,
    trailing tiles are clamped to align cleanly with the bottom/right edges so
    every pixel is covered without out-of-bounds reads.
    """
    if tile_size < 1:
        raise ValueError("tile_size must be >= 1.")
    step = stride if stride is not None else tile_size
    if step < 1:
        raise ValueError("stride must be >= 1.")

    def _axis_starts(total: int) -> list[int]:
        if total <= tile_size:
            return [0]
        starts = list(range(0, total - tile_size + 1, step))
        last = total - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    ys = _axis_starts(height)
    xs = _axis_starts(width)

    grid: list[TileSpec] = []
    for r_idx, y0 in enumerate(ys):
        y1 = min(y0 + tile_size, height)
        for c_idx, x0 in enumerate(xs):
            x1 = min(x0 + tile_size, width)
            grid.append(TileSpec(row=r_idx, col=c_idx, y0=y0, y1=y1, x0=x0, x1=x1))
    return grid


def pad_chw_to_min_size(
    array: np.ndarray,
    min_h: int,
    min_w: int,
    mode: str = "reflect",
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a ``(C, H, W)`` or ``(H, W)`` array if smaller than ``(min_h, min_w)``."""
    is_2d = array.ndim == 2
    if is_2d:
        h, w = array.shape
    elif array.ndim == 3:
        _, h, w = array.shape
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}.")

    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return array, (h, w)

    pad_mode = mode if (h > 1 and w > 1) else "constant"
    if is_2d:
        padded = np.pad(array, ((0, pad_h), (0, pad_w)), mode=pad_mode)
    else:
        padded = np.pad(array, ((0, 0), (0, pad_h), (0, pad_w)), mode=pad_mode)
    return padded, (h, w)


def extract_patches(
    array: np.ndarray,
    tile_size: int = 256,
    stride: int | None = None,
    pad_if_smaller: bool = True,
) -> tuple[list[np.ndarray], list[TileSpec], tuple[int, int]]:
    """Extract fixed-size ``tile_size x tile_size`` patches from ``(C, H, W)`` or ``(H, W)``."""
    is_2d = array.ndim == 2
    orig_h, orig_w = array.shape if is_2d else array.shape[1:]

    work = array
    if pad_if_smaller and (orig_h < tile_size or orig_w < tile_size):
        work, _ = pad_chw_to_min_size(array, tile_size, tile_size)

    work_h, work_w = work.shape if is_2d else work.shape[1:]
    specs = compute_tile_grid(work_h, work_w, tile_size=tile_size, stride=stride)
    patches: list[np.ndarray] = []
    for sp in specs:
        patch = work[sp.y0 : sp.y1, sp.x0 : sp.x1] if is_2d else work[:, sp.y0 : sp.y1, sp.x0 : sp.x1]
        patches.append(patch)
    return patches, specs, (orig_h, orig_w)


def create_blend_window(tile_size: int, mode: str = "hann", min_weight: float = 0.05) -> np.ndarray:
    """Generate a smooth 2D blending kernel ``(tile_size, tile_size)`` to avoid tile seams."""
    if mode == "uniform":
        return np.ones((tile_size, tile_size), dtype=np.float32)

    if mode == "hann":
        win_1d = np.hanning(tile_size).astype(np.float32)
        win_2d = np.outer(win_1d, win_1d)
        # Keep non-zero edge weights so non-overlapping borders still accumulate cleanly
        return np.clip(win_2d, min_weight, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported blend window mode '{mode}'. Choose 'hann' or 'uniform'.")


def stitch_tiles(
    tiles: list[np.ndarray],
    specs: list[TileSpec],
    out_height: int,
    out_width: int,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Blend predicted tiles back into a single ``(C, out_height, out_width)`` or ``(out_height, out_width)`` array."""
    if not tiles or len(tiles) != len(specs):
        raise ValueError("tiles and specs must be non-empty and have equal length.")

    first = tiles[0]
    is_2d = first.ndim == 2
    tile_h, tile_w = first.shape if is_2d else first.shape[1:]
    blend = window if window is not None else create_blend_window(tile_h, mode="hann")

    # Work buffer may be larger than (out_height, out_width) when input was padded up to tile_size
    buf_h = max(out_height, max(sp.y1 for sp in specs))
    buf_w = max(out_width, max(sp.x1 for sp in specs))

    if is_2d:
        acc = np.zeros((buf_h, buf_w), dtype=np.float32)
        w_acc = np.zeros((buf_h, buf_w), dtype=np.float32)
        for tile, sp in zip(tiles, specs):
            th, tw = sp.y1 - sp.y0, sp.x1 - sp.x0
            w_patch = blend[:th, :tw]
            acc[sp.y0 : sp.y1, sp.x0 : sp.x1] += tile[:th, :tw] * w_patch
            w_acc[sp.y0 : sp.y1, sp.x0 : sp.x1] += w_patch
        out = acc / np.maximum(w_acc, 1e-6)
        return out[:out_height, :out_width].astype(np.float32)

    channels = first.shape[0]
    acc = np.zeros((channels, buf_h, buf_w), dtype=np.float32)
    w_acc = np.zeros((1, buf_h, buf_w), dtype=np.float32)
    for tile, sp in zip(tiles, specs):
        th, tw = sp.y1 - sp.y0, sp.x1 - sp.x0
        w_patch = blend[:th, :tw][None, :, :]
        acc[:, sp.y0 : sp.y1, sp.x0 : sp.x1] += tile[:, :th, :tw] * w_patch
        w_acc[:, sp.y0 : sp.y1, sp.x0 : sp.x1] += w_patch
    out = acc / np.maximum(w_acc, 1e-6)
    return out[:, :out_height, :out_width].astype(np.float32)

