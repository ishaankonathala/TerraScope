"""Custom user-supplied satellite/RGB change-detection dataset and preparation utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from ml.datasets.transforms import PairedChangeTransform, PairedTransformConfig
from ml.preprocessing.alignment import align_image_pair
from ml.preprocessing.image_loader import (
    SUPPORTED_IMAGE_EXTENSIONS,
    ImageValidationError,
    load_image,
    load_mask,
)
from ml.preprocessing.tiling import extract_patches


class CustomPairDataset(Dataset):
    """Dataset for validated custom before/after image pairs (with optional masks).

    Supports either a directory layout (``before/``, ``after/``, optional ``mask/``)
    or an explicit list of ``(before_path, after_path, mask_path)`` records.
    """

    def __init__(
        self,
        pairs: list[tuple[Path, Path, Path | None]] | None = None,
        root_dir: str | Path | None = None,
        image_size: tuple[int, int] = (256, 256),
        expected_channels: int | None = 3,
        is_train: bool = False,
        normalize_mode: str = "imagenet",
        max_size_mb: float = 50.0,
        transform: PairedChangeTransform | None = None,
    ) -> None:
        self.expected_channels = expected_channels
        self.max_size_mb = max_size_mb

        if pairs is not None:
            self.pairs = [
                (Path(b), Path(a), Path(m) if m is not None else None) for b, a, m in pairs
            ]
        elif root_dir is not None:
            self.pairs = self._discover_pairs(Path(root_dir))
        else:
            raise ValueError("Provide either 'pairs' or 'root_dir' to CustomPairDataset.")

        if not self.pairs:
            raise FileNotFoundError("No valid before/after image pairs discovered.")

        if transform is not None:
            self.transform = transform
        else:
            cfg = PairedTransformConfig(output_size=image_size, normalize_mode=normalize_mode)
            self.transform = PairedChangeTransform(config=cfg, is_train=is_train)

    @staticmethod
    def _discover_pairs(root: Path) -> list[tuple[Path, Path, Path | None]]:
        before_dir = root / "before"
        after_dir = root / "after"
        mask_dir = root / "mask"
        if not (before_dir.is_dir() and after_dir.is_dir()):
            raise FileNotFoundError(
                f"Custom dataset under '{root}' must contain 'before/' and 'after/' folders."
            )

        after_by_stem = {
            p.stem: p
            for p in after_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        mask_by_stem = (
            {
                p.stem: p
                for p in mask_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            }
            if mask_dir.is_dir()
            else {}
        )

        pairs: list[tuple[Path, Path, Path | None]] = []
        for b_path in sorted(before_dir.iterdir()):
            if not (b_path.is_file() and b_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS):
                continue
            a_path = after_dir / b_path.name
            if not a_path.is_file():
                a_path = after_by_stem.get(b_path.stem)
            if a_path is None:
                raise ImageValidationError(
                    f"Missing matching 'after' image for '{b_path.name}' in '{after_dir}'."
                )
            m_path = mask_by_stem.get(b_path.stem)
            pairs.append((b_path, a_path, m_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        b_path, a_path, m_path = self.pairs[index]
        before = load_image(
            b_path, expected_channels=self.expected_channels, max_size_mb=self.max_size_mb
        )
        after = load_image(
            a_path, expected_channels=self.expected_channels, max_size_mb=self.max_size_mb
        )
        before_aligned, after_aligned, align_report = align_image_pair(before, after)

        mask_arr = load_mask(m_path, max_size_mb=self.max_size_mb) if m_path is not None else None
        if mask_arr is not None and mask_arr.shape != (
            before_aligned.height,
            before_aligned.width,
        ):
            import cv2

            mask_arr = cv2.resize(
                mask_arr,
                (before_aligned.width, before_aligned.height),
                interpolation=cv2.INTER_NEAREST,
            )

        out = self.transform(
            before_aligned.data,
            after_aligned.data,
            mask=mask_arr,
            valid_mask=before_aligned.valid_mask & after_aligned.valid_mask,
        )
        out["meta"] = {
            "dataset": "Custom",
            "sample_id": b_path.stem,
            "before_path": str(b_path),
            "after_path": str(a_path),
            "label_path": str(m_path) if m_path is not None else "",
            "alignment_method": align_report.method,
        }
        return out


def prepare_custom_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 256,
    expected_channels: int | None = 3,
    max_size_mb: float = 50.0,
) -> dict[str, Any]:
    """Validate, align, and tile a custom dataset directory into ``output_dir``.

    Saves patches as PNG (for 3-channel uint8 imagery) or GeoTIFF/NPY-safe TIFF
    plus a ``manifest.json`` summary so large user pairs can be trained cleanly.
    """
    src = Path(source_dir)
    dst = Path(output_dir)
    pairs = CustomPairDataset._discover_pairs(src)

    out_b = dst / "before"
    out_a = dst / "after"
    out_m = dst / "mask"
    out_b.mkdir(parents=True, exist_ok=True)
    out_a.mkdir(parents=True, exist_ok=True)
    has_any_mask = any(m is not None for _, _, m in pairs)
    if has_any_mask:
        out_m.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, Any]] = []
    for b_path, a_path, m_path in pairs:
        before = load_image(b_path, expected_channels=expected_channels, max_size_mb=max_size_mb)
        after = load_image(a_path, expected_channels=expected_channels, max_size_mb=max_size_mb)
        before_al, after_al, align_rep = align_image_pair(before, after)

        b_patches, specs, _ = extract_patches(before_al.data, tile_size=patch_size, stride=stride)
        a_patches, _, _ = extract_patches(after_al.data, tile_size=patch_size, stride=stride)

        m_patches = None
        if m_path is not None:
            mask = load_mask(m_path, max_size_mb=max_size_mb)
            if mask.shape != (before_al.height, before_al.width):
                import cv2

                mask = cv2.resize(
                    mask, (before_al.width, before_al.height), interpolation=cv2.INTER_NEAREST
                )
            m_patches, _, _ = extract_patches(mask, tile_size=patch_size, stride=stride)

        for idx, (bp, ap, sp) in enumerate(zip(b_patches, a_patches, specs)):
            tile_name = f"{b_path.stem}_r{sp.row}_c{sp.col}.png"
            bp_u8 = np.clip(np.transpose(bp, (1, 2, 0)), 0, 255).astype(np.uint8)
            ap_u8 = np.clip(np.transpose(ap, (1, 2, 0)), 0, 255).astype(np.uint8)
            Image.fromarray(bp_u8).save(out_b / tile_name)
            Image.fromarray(ap_u8).save(out_a / tile_name)

            mask_out_path = None
            if m_patches is not None:
                mp_u8 = (m_patches[idx] > 0).astype(np.uint8) * 255
                mask_out_path = str(out_m / tile_name)
                Image.fromarray(mp_u8, mode="L").save(out_m / tile_name)

            manifest_entries.append(
                {
                    "id": f"{b_path.stem}_r{sp.row}_c{sp.col}",
                    "before": str(out_b / tile_name),
                    "after": str(out_a / tile_name),
                    "mask": mask_out_path,
                    "alignment": align_rep.method,
                }
            )

    summary = {
        "source_dir": str(src),
        "output_dir": str(dst),
        "source_pairs": len(pairs),
        "generated_patches": len(manifest_entries),
        "patch_size": patch_size,
        "stride": stride,
        "patches": manifest_entries,
    }
    (dst / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

