"""OSCD (Onera Satellite Change Detection) multispectral Sentinel-2 dataset loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import rasterio
from torch.utils.data import Dataset

from ml.datasets.transforms import PairedChangeTransform, PairedTransformConfig
from ml.preprocessing.image_loader import (
    LoadedImage,
    load_image,
    load_multispectral_stack,
)
from ml.preprocessing.tiling import TileSpec, compute_tile_grid

OSCD_URL = "https://rcdaudt.github.io/oscd/"

# Official OSCD city splits (Daudt et al., IGARSS 2018; train.txt / test.txt)
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

BAND_ALIASES: dict[str, tuple[str, ...]] = {
    "B01": ("B01", "B1"),
    "B02": ("B02", "B2"),
    "B03": ("B03", "B3"),
    "B04": ("B04", "B4"),
    "B05": ("B05", "B5"),
    "B06": ("B06", "B6"),
    "B07": ("B07", "B7"),
    "B08": ("B08", "B8"),
    "B8A": ("B8A", "B08A", "B8"),
    "B09": ("B09", "B9"),
    "B10": ("B10",),
    "B11": ("B11",),
    "B12": ("B12",),
}


def load_oscd_mask(path: str | Path) -> np.ndarray:
    """Load an OSCD ground-truth change mask and convert to binary uint8 {0, 1}.

    In the official OSCD benchmark (Daudt et al., IGARSS 2018):
    - TIFF masks (`<city>-cm.tif` or `cm.tif`) use integer class labels:
        1 = No change (background)
        2 = Change (urban/land-use change)
        0 = Nodata / unannotated
    - PNG masks (`cm.png`) typically use:
        0 = No change
        255 = Change (or 0 and 1, or 1 and 2)

    This function strictly maps change pixels to 1 and unchanged/nodata pixels to 0.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"OSCD ground-truth mask file not found: {p}")

    ext = p.suffix.lower()
    if ext in {".tif", ".tiff"}:
        with rasterio.open(p) as src:
            arr = src.read(1)
    else:
        with Image.open(p) as img:
            arr = np.asarray(img.convert("L"))

    unique_vals = set(np.unique(arr))

    # Official OSCD {1: no-change, 2: change, 0: nodata}
    if 2 in unique_vals and unique_vals.issubset({0, 1, 2}):
        return (arr == 2).astype(np.uint8)

    # Standard binary mask {0: no-change, 255: change}
    if 255 in unique_vals:
        return (arr == 255).astype(np.uint8)

    # Fallback binary {0: no-change, 1: change}
    return (arr > 0).astype(np.uint8)


def _get_raster_dimensions(path: Path) -> tuple[int, int]:
    """Inspect (height, width) from file header without full raster decoding."""
    ext = path.suffix.lower()
    if ext in {".tif", ".tiff"}:
        with rasterio.open(path) as src:
            return int(src.height), int(src.width)
    with Image.open(path) as img:
        return int(img.height), int(img.width)


class OSCDDataset(Dataset):
    """PyTorch Dataset for the Onera Satellite Change Detection (OSCD) benchmark.

    Supports:
    - Official multi-directory layout:
        Onera Satellite Change Detection dataset - Images/
        Onera Satellite Change Detection dataset - Train Labels/
        Onera Satellite Change Detection dataset - Test Labels/
    - Unified single-directory layout (labels inside city/cm/ or labels/ folder).
    - Multi-band Sentinel-2 GeoTIFFs (RGB, 4-band RGB-NIR, or full 13-band stack).
    - Preference for `<city>-cm.tif` with `cm.png` fallback.
    - Strict validation of raster-mask dimensions (no silent resizing).
    - Strict enforcement of official train (14 cities) and test (10 cities) splits.
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
        train_labels_dir: str | Path | None = None,
        test_labels_dir: str | Path | None = None,
        strict: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.split = split.lower().strip()
        if self.split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported OSCD split '{split}'. Choose from {self.VALID_SPLITS}.")

        self.bands = tuple(b.upper() for b in bands)
        self.patch_size = patch_size
        self.stride = stride
        self.strict = strict

        if not self.root_dir.is_dir():
            raise FileNotFoundError(
                f"OSCD dataset directory not found at '{self.root_dir}'. "
                f"Download from {OSCD_URL} according to its CC-BY-4.0 license."
            )

        # 1. Discover the base directory containing city image folders
        self.images_base = self._find_images_base(self.root_dir)
        city_dirs = {p.name.lower().strip(): p for p in self.images_base.iterdir() if p.is_dir()}

        # 2. Discover train and test label directories
        self.train_label_dirs = self._find_label_roots(
            explicit_dir=train_labels_dir,
            category="train",
        )
        self.test_label_dirs = self._find_label_roots(
            explicit_dir=test_labels_dir,
            category="test",
        )

        # 3. Resolve target cities for the requested split
        target_cities = self._resolve_split_cities(city_dirs, val_cities)
        if not target_cities:
            raise FileNotFoundError(
                f"No OSCD city directories found matching split '{self.split}' under '{self.images_base}'. "
                f"Refer to {OSCD_URL} for official folder conventions."
            )

        # 4. Build samples and validate dimensions
        self.samples: list[dict[str, Any]] = []
        for city_name in target_cities:
            cdir = city_dirs[city_name]
            t1_dir = self._find_temporal_dir(cdir, ("imgs_1_rect", "imgs_1"))
            t2_dir = self._find_temporal_dir(cdir, ("imgs_2_rect", "imgs_2"))

            label_path = self._find_label_file(cdir, city_name, required=self.strict)
            if label_path is None or t1_dir is None or t2_dir is None:
                continue

            t1_paths, resolved_bands = self._resolve_band_paths(t1_dir)
            t2_paths, _ = self._resolve_band_paths(t2_dir)
            if not t1_paths or len(t1_paths) != len(t2_paths):
                continue

            # Verify dimension alignment: mask MUST match rectified satellite images
            ref_band_path = t1_paths[0]
            img_h, img_w = _get_raster_dimensions(ref_band_path)
            mask_h, mask_w = _get_raster_dimensions(label_path)
            if (img_h, img_w) != (mask_h, mask_w):
                raise ValueError(
                    f"OSCD spatial alignment error for city '{city_name}': "
                    f"Satellite image dimensions ({img_h}, {img_w}) from '{ref_band_path.name}' "
                    f"do not match ground-truth mask dimensions ({mask_h}, {mask_w}) from '{label_path.name}'. "
                    f"Silent resizing is disabled to prevent spatial misalignment."
                )

            grid = compute_tile_grid(mask_h, mask_w, tile_size=self.patch_size, stride=self.stride)
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
                f"Found OSCD city folders under '{self.images_base}', but none contained complete "
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

    @classmethod
    def _find_images_base(cls, root_dir: Path) -> Path:
        """Find the folder containing individual city image directories."""
        all_cities = set(OSCD_TRAIN_CITIES) | set(OSCD_TEST_CITIES)
        child_dirs = {p.name.lower().strip(): p for p in root_dir.iterdir() if p.is_dir()}
        if any(c in all_cities for c in child_dirs):
            return root_dir

        for candidate_name in (
            "Onera Satellite Change Detection dataset - Images",
            "images",
            "Images",
            "imgs",
        ):
            cand = root_dir / candidate_name
            if cand.is_dir():
                return cand

        # Fuzzy matching: child folder containing "image"
        for p in root_dir.iterdir():
            if p.is_dir() and "image" in p.name.lower():
                return p

        # Check parent folder if root_dir itself was pointed at a subfolder
        if root_dir.parent != root_dir:
            for p in root_dir.parent.iterdir():
                if p.is_dir() and "image" in p.name.lower():
                    return p

        return root_dir

    def _find_label_roots(
        self,
        explicit_dir: str | Path | None,
        category: str,
    ) -> list[Path]:
        """Find candidate roots containing city label directories for train or test."""
        discovered: list[Path] = []
        if explicit_dir is not None:
            p = Path(explicit_dir).resolve()
            if p.is_dir():
                discovered.append(p)

        search_bases = [self.root_dir]
        if self.root_dir.parent != self.root_dir and self.root_dir.parent not in search_bases:
            search_bases.append(self.root_dir.parent)
        if hasattr(self, "images_base") and self.images_base.parent not in search_bases:
            search_bases.append(self.images_base.parent)

        target_tokens = ("train", "label") if category == "train" else ("test", "label")
        standard_names = (
            f"Onera Satellite Change Detection dataset - {category.title()} Labels",
            f"{category}_labels",
            f"{category.title()} Labels",
            f"{category.title()}_Labels",
            f"{category}_label",
            category,
        )

        for base in search_bases:
            if not base.is_dir():
                continue
            for name in standard_names:
                cand = base / name
                if cand.is_dir() and cand not in discovered:
                    discovered.append(cand)

            # Subdirectory search matching both tokens (e.g. "train" and "label")
            for sub in base.iterdir():
                if sub.is_dir() and sub not in discovered:
                    sub_lower = sub.name.lower()
                    if all(token in sub_lower for token in target_tokens):
                        discovered.append(sub)

            # Generic labels directory fallback
            for gen_name in ("labels", "Labels"):
                cand = base / gen_name
                if cand.is_dir() and cand not in discovered:
                    discovered.append(cand)

        return discovered

    def _resolve_split_cities(
        self, city_dirs: dict[str, Path], val_cities: Sequence[str]
    ) -> list[str]:
        """Resolve city list ensuring strict adherence to official OSCD train/test split."""
        val_set = {c.lower().strip() for c in val_cities}

        # Enforce Requirement 5: Validation cities must only come from the official training split
        invalid_val = val_set - set(OSCD_TRAIN_CITIES)
        if invalid_val:
            raise ValueError(
                f"Validation cities {sorted(invalid_val)} are not in the official OSCD training split. "
                f"Validation must only use cities from OSCD_TRAIN_CITIES ({len(OSCD_TRAIN_CITIES)} cities). "
                f"Training and testing cities must never be mixed."
            )

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

    def _find_label_file(
        self, city_dir: Path, city_name: str, required: bool = False
    ) -> Path | None:
        """Locate ground-truth change mask preferring <city>-cm.tif over cm.png.

        Checks:
        1. Separate official Train/Test label directories
        2. City directory cm/ folder and city directory root (existing layout fallback)
        """
        norm_city = city_name.lower().strip()

        # Decide which label directory list to prioritize
        if norm_city in OSCD_TRAIN_CITIES:
            label_dirs = list(self.train_label_dirs) + list(self.test_label_dirs)
        elif norm_city in OSCD_TEST_CITIES:
            label_dirs = list(self.test_label_dirs) + list(self.train_label_dirs)
        else:
            label_dirs = list(self.train_label_dirs) + list(self.test_label_dirs)

        # 1. Search in separate official label directories
        for ldir in label_dirs:
            if not ldir.is_dir():
                continue
            matching_subdirs = [p for p in ldir.iterdir() if p.is_dir() and p.name.lower().strip() == norm_city]
            if matching_subdirs:
                city_label_dir = matching_subdirs[0]
                label_cand = self._pick_label_from_city_folder(city_label_dir, norm_city)
                if label_cand is not None:
                    return label_cand

        # 2. Search in city_dir itself (existing layout fallback)
        label_cand = self._pick_label_from_city_folder(city_dir, norm_city)
        if label_cand is not None:
            return label_cand

        if required:
            raise FileNotFoundError(
                f"Ground-truth change mask for OSCD city '{city_name}' not found. "
                f"Looked for '{norm_city}-cm.tif' (and 'cm.png' fallback) in separate label "
                f"directories and inside '{city_dir}'."
            )
        return None

    @staticmethod
    def _pick_label_from_city_folder(city_folder: Path, city_name: str) -> Path | None:
        """Select label file preferring <city>-cm.tif > cm.tif > cm.png inside a city label folder."""
        cm_sub = city_folder / "cm"
        folders_to_check = [cm_sub, city_folder] if cm_sub.is_dir() else [city_folder]

        # Preference 1: TIFF masks (<city>-cm.tif, cm.tif)
        tiff_names = (
            f"{city_name}-cm.tif",
            f"{city_name}-cm.tiff",
            f"{city_folder.name.lower()}-cm.tif",
            f"{city_folder.name.lower()}-cm.tiff",
            "cm.tif",
            "cm.tiff",
        )
        for folder in folders_to_check:
            for name in tiff_names:
                candidate = folder / name
                if candidate.is_file():
                    return candidate

        # Preference 2: PNG masks (cm.png, <city>-cm.png fallback)
        png_names = (
            "cm.png",
            f"{city_name}-cm.png",
            f"{city_folder.name.lower()}-cm.png",
        )
        for folder in folders_to_check:
            for name in png_names:
                candidate = folder / name
                if candidate.is_file():
                    return candidate

        # Fallback 3: Case-insensitive scan for any file matching *cm*.tif* then *cm*.png*
        for folder in folders_to_check:
            for f in folder.iterdir():
                if f.is_file() and "cm" in f.name.lower() and f.suffix.lower() in {".tif", ".tiff"}:
                    return f
            for f in folder.iterdir():
                if f.is_file() and "cm" in f.name.lower() and f.suffix.lower() == ".png":
                    return f

        return None

    def _resolve_band_paths(self, img_dir: Path) -> tuple[list[Path], list[str]]:
        available = {p.stem.upper(): p for p in img_dir.iterdir() if p.is_file()}
        paths: list[Path] = []
        names: list[str] = []

        for band in self.bands:
            band_up = band.upper()
            cand_keys = BAND_ALIASES.get(band_up, (band_up,))
            matched_key = next((k for k in cand_keys if k in available), None)
            if matched_key is not None:
                paths.append(available[matched_key])
                names.append(band_up)

        if len(paths) == len(self.bands):
            return paths, names

        # Fallback: single pre-rendered RGB composite image inside folder
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
        mask = load_oscd_mask(spec["label_path"])

        # Spatial dimension integrity check (no silent resizing)
        if (t1_loaded.height, t1_loaded.width) != mask.shape:
            raise ValueError(
                f"OSCD dimension mismatch for city '{spec['city']}': "
                f"T1 dimensions ({t1_loaded.height}, {t1_loaded.width}) != mask shape {mask.shape}."
            )
        if (t2_loaded.height, t2_loaded.width) != mask.shape:
            raise ValueError(
                f"OSCD dimension mismatch for city '{spec['city']}': "
                f"T2 dimensions ({t2_loaded.height}, {t2_loaded.width}) != mask shape {mask.shape}."
            )

        # Extract patch tiles
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
