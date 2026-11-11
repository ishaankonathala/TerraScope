"""OSCD (Onera Satellite Change Detection) multispectral Sentinel-2 dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from torch.utils.data import Dataset

from ml.datasets.transforms import PairedChangeTransform, PairedTransformConfig
from ml.preprocessing.image_loader import (
    LoadedImage,
    load_image,
    load_mask,
    load_multispectral_stack,
)
from ml.preprocessing.tiling import TileSpec, compute_tile_grid

OSCD_URL = "https://rcdaudt.github.io/oscd/"

# Official OSCD city splits (Daudt et al., IGARSS 2018)
OSCD_TRAIN_CITIES: tuple[str, ...] = (
    "abudhabi",
    "aguasclaras",
    "beihai",
    "beirut",
    "bercy",
    "bordeaux",
    "cupertino",
    "hongkong",
    "mumbai",
    "nantes",
    "paris",
    "pisa",
    "rennes",
    "saclay_e",
)

OSCD_TEST_CITIES: tuple[str, ...] = (
    "brasilia",
    "chongqing",
    "dubai",
    "lasvegas",
    "milano",
    "montpellier",
    "norcia",
    "rio",
    "saclay_w",
    "valencia",
)

SENTINEL2_13_BANDS: tuple[str, ...] = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
)


class OSCDDataset(Dataset):
    """PyTorch Dataset for the Onera Satellite Change Detection (OSCD) benchmark.

    Supports multi-band Sentinel-2 GeoTIFF inputs (e.g. RGB ``(B04, B03, B02)``,
    4-band ``(B02, B03, B04, B08)``, or all 13 Sentinel-2 bands) as well as
    pre-rendered RGB pairs, preserving geospatial metadata where available and
    extracting fixed-size patches from large urban scenes.
    """

    VALID_SPLITS = ("train", "val", "test", "all")

    def __init__(
        self,
        root_dir: str | Path,
        split: str = "train",
        bands: Sequence[str] = ("B04", "B03", "B02"),
        patch_size: int = 256,
        stride: int = 256,
        val_cities: Sequence[str] = ("paris", "pisa", "mumbai"),
        transform: PairedChangeTransform | None = None,
        normalize_mode: str = "sentinel2_sr",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split.lower()
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported OSCD split '{split}'. Choose from {self.VALID_SPLITS}.")

        self.bands = tuple(b.upper() for b in bands)
        self.patch_size = patch_size
        self.stride = stride

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"OSCD dataset directory not found at '{self.root_dir}'. "
                f"Download from {OSCD_URL} according to its CC-BY-4.0 license."
            )

        images_base = self.root_dir / "images" if (self.root_dir / "images").is_dir() else self.root_dir
        city_dirs = {p.name.lower(): p for p in images_base.iterdir() if p.is_dir()}

        target_cities = self._resolve_split_cities(city_dirs, val_cities)
        if not target_cities:
            raise FileNotFoundError(
                f"No OSCD city directories found matching split '{self.split}' under '{images_base}'. "
                f"Refer to {OSCD_URL} for official folder conventions."
            )

        self.samples: list[dict[str, Any]] = []
        for city_name in target_cities:
            cdir = city_dirs[city_name]
            t1_dir = self._find_temporal_dir(cdir, ("imgs_1_rect", "imgs_1"))
            t2_dir = self._find_temporal_dir(cdir, ("imgs_2_rect", "imgs_2"))
            label_path = self._find_label_file(cdir, city_name)

            if t1_dir is None or t2_dir is None or label_path is None:
                continue

            t1_paths, resolved_bands = self._resolve_band_paths(t1_dir)
            t2_paths, _ = self._resolve_band_paths(t2_dir)
            if not t1_paths or len(t1_paths) != len(t2_paths):
                continue

            mask_arr = load_mask(label_path)
            h, w = mask_arr.shape
            grid = compute_tile_grid(h, w, tile_size=self.patch_size, stride=self.stride)
            for tile in grid:
                self.samples.append(
                    {
                        "city": city_name,
                        "t1_paths": t1_paths,
                        "t2_paths": t2_paths,
                        "bands": resolved_bands,
                        "label_path": label_path,
                        "tile": tile,
                    }
                )

        if not self.samples:
            raise FileNotFoundError(
                f"Found OSCD city folders under '{images_base}', but none contained complete "
                f"temporal bands ({self.bands}) and change masks."
            )

        if transform is not None:
            self.transform = transform
        else:
            is_train = self.split == "train"
            cfg = PairedTransformConfig(
                output_size=(self.patch_size, self.patch_size),
                normalize_mode=normalize_mode,
            )
            self.transform = PairedChangeTransform(config=cfg, is_train=is_train)

    def _resolve_split_cities(
        self, city_dirs: dict[str, Path], val_cities: Sequence[str]
    ) -> list[str]:
        val_set = {c.lower() for c in val_cities}
        if self.split == "all":
            return sorted(city_dirs.keys())
        if self.split == "test":
            return sorted(c for c in city_dirs if c in OSCD_TEST_CITIES)
        if self.split == "val":
            return sorted(c for c in city_dirs if c in OSCD_TRAIN_CITIES and c in val_set)
        # train = official train cities excluding validation subset
        return sorted(c for c in city_dirs if c in OSCD_TRAIN_CITIES and c not in val_set)

    @staticmethod
    def _find_temporal_dir(city_dir: Path, candidates: tuple[str, ...]) -> Path | None:
        for name in candidates:
            cand = city_dir / name
            if cand.is_dir():
                return cand
        return None

    def _find_label_file(self, city_dir: Path, city_name: str) -> Path | None:
        direct_candidates = [
            city_dir / "cm" / "cm.png",
            city_dir / "cm" / f"{city_name}-cm.tif",
            city_dir / f"{city_name}-cm.tif",
            city_dir / "cm.png",
            self.root_dir / "labels" / city_name / "cm" / "cm.png",
            self.root_dir / "train_labels" / city_name / "cm" / "cm.png",
            self.root_dir / "test_labels" / city_name / "cm" / "cm.png",
        ]
        for cand in direct_candidates:
            if cand.is_file():
                return cand
        return None

    def _resolve_band_paths(self, img_dir: Path) -> tuple[list[Path], list[str]]:
        available = {p.stem.upper(): p for p in img_dir.iterdir() if p.is_file()}
        paths: list[Path] = []
        names: list[str] = []
        for band in self.bands:
            if band in available:
                paths.append(available[band])
                names.append(band)
        if len(paths) == len(self.bands):
            return paths, names

        # Fallback: single RGB composite image inside folder if individual GeoTIFF bands aren't unpacked
        for ext_name in ("img.png", "rgb.png", "rgb.tif"):
            p = img_dir / ext_name
            if p.is_file() and len(self.bands) == 3:
                return [p], ["RGB"]
        return [], []

    @staticmethod
    def _load_temporal_image(paths: list[Path], bands: list[str]) -> LoadedImage:
        if len(paths) == 1 and bands == ["RGB"]:
            return load_image(paths[0], expected_channels=3)
        # Prefer a 10m band (B02, B03, B04, B08) as spatial reference grid
        ref_idx = 0
        for idx, b in enumerate(bands):
            if b in {"B04", "B03", "B02", "B08"}:
                ref_idx = idx
                break
        return load_multispectral_stack(paths, band_names=bands, reference_index=ref_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        spec = self.samples[index]
        tile: TileSpec = spec["tile"]

        t1_loaded = self._load_temporal_image(spec["t1_paths"], spec["bands"])
        t2_loaded = self._load_temporal_image(spec["t2_paths"], spec["bands"])
        mask = load_mask(spec["label_path"])

        # Handle patch near edges if smaller than patch_size
        b_patch = t1_loaded.data[:, tile.y0 : tile.y1, tile.x0 : tile.x1]
        a_patch = t2_loaded.data[:, tile.y0 : tile.y1, tile.x0 : tile.x1]
        m_patch = mask[tile.y0 : tile.y1, tile.x0 : tile.x1]
        v_patch = (t1_loaded.valid_mask & t2_loaded.valid_mask)[
            tile.y0 : tile.y1, tile.x0 : tile.x1
        ]

        out = self.transform(b_patch, a_patch, mask=m_patch, valid_mask=v_patch)
        out["meta"] = {
            "dataset": "OSCD",
            "split": self.split,
            "city": spec["city"],
            "sample_id": f"{spec['city']}_r{tile.row}_c{tile.col}",
            "bands": spec["bands"],
            "geospatial": t1_loaded.metadata.to_dict(),
        }
        return out

