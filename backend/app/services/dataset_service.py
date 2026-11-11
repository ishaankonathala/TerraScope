"""Dataset inspection service reporting factual local split availability."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.app.core.config import Settings
from ml.datasets import LEVIR_CD_URL, OSCD_URL
from ml.preprocessing.image_loader import SUPPORTED_IMAGE_EXTENSIONS, load_image


class DatasetService:
    """Inspects real dataset directories without inventing synthetic data."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _inspect_levir(self, levir_dir: Path) -> dict[str, Any]:
        splits: dict[str, Any] = {}
        total = 0
        for sp in ("train", "val", "test"):
            a_dir = levir_dir / sp / "A"
            b_dir = levir_dir / sp / "B"
            l_dir = levir_dir / sp / "label"
            if a_dir.is_dir() and b_dir.is_dir() and l_dir.is_dir():
                files = [
                    p
                    for p in sorted(a_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ]
                sample_dims = None
                if files:
                    try:
                        img = load_image(files[0])
                        sample_dims = f"{img.width}x{img.height}x{img.channels}"
                    except Exception:
                        sample_dims = None
                splits[sp] = {"count": len(files), "sample_dimensions": sample_dims}
                total += len(files)
            else:
                splits[sp] = {"count": 0, "sample_dimensions": None}

        return {
            "id": "levir_cd",
            "name": "LEVIR-CD",
            "purpose": "High-resolution building change detection (RGB)",
            "official_url": LEVIR_CD_URL,
            "available": total > 0,
            "total_images": total,
            "bands": ["R", "G", "B"],
            "splits": splits,
            "path": "data/raw/LEVIR-CD",
        }

    def _inspect_oscd(self, oscd_dir: Path) -> dict[str, Any]:
        images_dir = oscd_dir / "images" if (oscd_dir / "images").is_dir() else oscd_dir
        cities = (
            [p.name for p in sorted(images_dir.iterdir()) if p.is_dir() and p.name not in {"labels", "train_labels", "test_labels"}]
            if images_dir.is_dir()
            else []
        )
        return {
            "id": "oscd",
            "name": "OSCD (Onera Satellite Change Detection)",
            "purpose": "Multi-temporal Sentinel-2 urban change detection",
            "official_url": OSCD_URL,
            "available": len(cities) > 0,
            "city_count": len(cities),
            "cities": cities,
            "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"],
            "path": "data/raw/OSCD",
        }

    def _inspect_samples(self, samples_dir: Path) -> dict[str, Any]:
        pairs: list[dict[str, Any]] = []
        if samples_dir.is_dir():
            before_dir = samples_dir / "before"
            after_dir = samples_dir / "after"
            if before_dir.is_dir() and after_dir.is_dir():
                for p in sorted(before_dir.iterdir()):
                    if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                        if (after_dir / p.name).is_file():
                            pairs.append({"name": p.name})
        return {
            "id": "custom_samples",
            "name": "Custom / GEE Sample Pairs",
            "purpose": "User-supplied pairs and Google Earth Engine GeoTIFF exports",
            "available": len(pairs) > 0,
            "pair_count": len(pairs),
            "pairs": pairs,
            "path": "data/samples",
        }

    def list_datasets(self) -> dict[str, Any]:
        raw_dir = self.settings.data_path / "raw"
        return {
            "datasets": [
                self._inspect_levir(raw_dir / "LEVIR-CD"),
                self._inspect_oscd(raw_dir / "OSCD"),
                self._inspect_samples(self.settings.data_path / "samples"),
            ]
        }

