"""LEVIR-CD building change-detection dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from ml.datasets.transforms import PairedChangeTransform, PairedTransformConfig
from ml.preprocessing.image_loader import SUPPORTED_IMAGE_EXTENSIONS, load_image, load_mask
from ml.preprocessing.tiling import TileSpec, compute_tile_grid

LEVIR_CD_URL = "https://justchenhao.github.io/LEVIR/"


class LEVIRCDDataset(Dataset):
    """PyTorch Dataset for LEVIR-CD multi-temporal building change detection.

    Supports ``train``, ``val``, and ``test`` splits with standard ``A/``, ``B/``,
    and ``label/`` directory structure. Large images (typically 1024x1024) can
    either be resized to ``image_size`` or sliced into non-overlapping/overlapping
    ``patch_size`` tiles for memory-efficient training on Apple Silicon or GPUs.
    """

    VALID_SPLITS = ("train", "val", "test")

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        patch_size: int | None = 256,
        stride: int | None = 256,
        image_size: tuple[int, int] = (256, 256),
        transform: PairedChangeTransform | None = None,
        normalize_mode: str = "imagenet",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported LEVIR-CD split '{split}'. Choose from {self.VALID_SPLITS}.")

        self.patch_size = patch_size
        self.stride = stride or patch_size

        split_dir = self.root_dir / self.split
        dir_a = split_dir / "A"
        dir_b = split_dir / "B"
        dir_lbl = split_dir / "label"

        if not (dir_a.is_dir() and dir_b.is_dir() and dir_lbl.is_dir()):
            raise FileNotFoundError(
                f"LEVIR-CD split '{self.split}' directory structure not found under '{split_dir}'. "
                f"Expected subdirectories A/, B/, and label/. "
                f"Please obtain LEVIR-CD from {LEVIR_CD_URL} according to its academic license."
            )

        self.samples: list[tuple[Path, Path, Path, TileSpec | None]] = []
        a_files = sorted(
            p for p in dir_a.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        if not a_files:
            raise FileNotFoundError(f"No valid image files found in '{dir_a}'.")

        b_lookup = {p.stem: p for p in dir_b.iterdir() if p.is_file()}
        lbl_lookup = {p.stem: p for p in dir_lbl.iterdir() if p.is_file()}

        for a_path in a_files:
            b_path = dir_b / a_path.name if (dir_b / a_path.name).exists() else b_lookup.get(a_path.stem)
            l_path = (
                dir_lbl / a_path.name
                if (dir_lbl / a_path.name).exists()
                else lbl_lookup.get(a_path.stem)
            )
            if b_path is None or l_path is None:
                raise FileNotFoundError(
                    f"Missing matching after/label file for LEVIR-CD sample '{a_path.name}'."
                )

            if self.patch_size is not None:
                # Inspect dimensions lightly via PIL/rasterio without loading whole batch into RAM
                img_meta = load_image(a_path, expected_channels=3).metadata
                grid = compute_tile_grid(
                    img_meta.height,
                    img_meta.width,
                    tile_size=self.patch_size,
                    stride=self.stride,
                )
                for tile in grid:
                    self.samples.append((a_path, b_path, l_path, tile))
            else:
                self.samples.append((a_path, b_path, l_path, None))

        if transform is not None:
            self.transform = transform
        else:
            is_train = self.split == "train"
            target_hw = (self.patch_size, self.patch_size) if self.patch_size else image_size
            cfg = PairedTransformConfig(output_size=target_hw, normalize_mode=normalize_mode)
            self.transform = PairedChangeTransform(config=cfg, is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        a_path, b_path, l_path, tile = self.samples[index]
        before_loaded = load_image(a_path, expected_channels=3)
        after_loaded = load_image(b_path, expected_channels=3)
        mask = load_mask(l_path)

        b_data = before_loaded.data
        a_data = after_loaded.data
        valid = before_loaded.valid_mask & after_loaded.valid_mask

        if b_data.shape[1:] != a_data.shape[1:] or b_data.shape[1:] != mask.shape:
            raise ValueError(
                f"Dimension mismatch in LEVIR-CD sample '{a_path.name}': "
                f"A={b_data.shape[1:]}, B={a_data.shape[1:]}, label={mask.shape}."
            )

        if tile is not None:
            b_data = b_data[:, tile.y0 : tile.y1, tile.x0 : tile.x1]
            a_data = a_data[:, tile.y0 : tile.y1, tile.x0 : tile.x1]
            mask = mask[tile.y0 : tile.y1, tile.x0 : tile.x1]
            valid = valid[tile.y0 : tile.y1, tile.x0 : tile.x1]

        out = self.transform(b_data, a_data, mask=mask, valid_mask=valid)
        out["meta"] = {
            "dataset": "LEVIR-CD",
            "split": self.split,
            "sample_id": a_path.stem
            if tile is None
            else f"{a_path.stem}_r{tile.row}_c{tile.col}",
            "before_path": str(a_path),
            "after_path": str(b_path),
            "label_path": str(l_path),
        }
        return out

