"""Comprehensive unit tests for OSCD dataset loader, split validation, and mask handling."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from PIL import Image
import rasterio
from rasterio.transform import from_origin

from ml.datasets.oscd import (
    OSCD_TEST_CITIES,
    OSCD_TRAIN_CITIES,
    OSCDDataset,
    load_oscd_mask,
)


def _create_mock_band(path: Path, height: int = 64, width: int = 64, val: float = 1000.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tf = from_origin(300000.0, 2780000.0, 10.0, 10.0)
    data = np.full((1, height, width), val, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32640",
        transform=tf,
    ) as dst:
        dst.write(data)


def _create_mock_mask_tiff(path: Path, height: int = 64, width: int = 64, mode: str = "oscd") -> None:
    """Create TIFF mask: mode 'oscd' has 1 (no-change) and 2 (change); mode 'binary' has 0 and 1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tf = from_origin(300000.0, 2780000.0, 10.0, 10.0)
    if mode == "oscd":
        # 1 = no change, 2 = change
        arr = np.ones((height, width), dtype=np.uint8)
        arr[: height // 2, :] = 2
    else:
        arr = np.zeros((height, width), dtype=np.uint8)
        arr[: height // 2, :] = 1

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs="EPSG:32640",
        transform=tf,
    ) as dst:
        dst.write(arr, 1)


def _create_mock_mask_png(path: Path, height: int = 64, width: int = 64, max_val: int = 255) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[: height // 2, :] = max_val
    Image.fromarray(arr).save(path)


def _setup_mock_city_images(base_dir: Path, city: str, height: int = 64, width: int = 64) -> Path:
    city_dir = base_dir / city
    for folder in ("imgs_1_rect", "imgs_2_rect"):
        _create_mock_band(city_dir / folder / "B04.tif", height, width, 1200.0)
        _create_mock_band(city_dir / folder / "B03.tif", height, width, 1000.0)
        _create_mock_band(city_dir / folder / "B02.tif", height, width, 800.0)
    return city_dir


# ==============================================================================
# 1. Test official separate train and test label directories discovery
# ==============================================================================

def test_oscd_separate_directories_discovery(tmp_path: Path) -> None:
    """Verify discovery when images, train labels, and test labels are in separate folders."""
    onera_root = tmp_path / "onera DATA"
    images_dir = onera_root / "Onera Satellite Change Detection dataset - Images"
    train_labels_dir = onera_root / "Onera Satellite Change Detection dataset - Train Labels"
    test_labels_dir = onera_root / "Onera Satellite Change Detection dataset - Test Labels"

    # Setup train city (paris) and test city (valencia)
    _setup_mock_city_images(images_dir, "paris")
    _setup_mock_city_images(images_dir, "valencia")

    # Put labels in respective official directories
    _create_mock_mask_tiff(train_labels_dir / "paris" / "cm" / "paris-cm.tif")
    _create_mock_mask_tiff(test_labels_dir / "valencia" / "cm" / "valencia-cm.tif")

    # A. Pointing to parent root
    ds_train = OSCDDataset(onera_root, split="train", val_cities=(), patch_size=64, stride=64)
    assert len(ds_train) == 1
    assert ds_train[0]["meta"]["city"] == "paris"

    ds_test = OSCDDataset(onera_root, split="test", patch_size=64, stride=64)
    assert len(ds_test) == 1
    assert ds_test[0]["meta"]["city"] == "valencia"

    # B. Pointing to Images subfolder directly (finds sibling labels via parent)
    ds_train_sub = OSCDDataset(images_dir, split="train", val_cities=(), patch_size=64, stride=64)
    assert len(ds_train_sub) == 1
    assert ds_train_sub[0]["meta"]["city"] == "paris"


# ==============================================================================
# 2. Test TIFF preference and PNG fallback
# ==============================================================================

def test_oscd_tiff_preference_and_png_fallback(tmp_path: Path) -> None:
    onera_root = tmp_path / "onera DATA"
    images_dir = onera_root / "Onera Satellite Change Detection dataset - Images"
    train_labels_dir = onera_root / "Onera Satellite Change Detection dataset - Train Labels"

    # City 1: Both TIFF and PNG exist -> TIFF MUST be preferred
    _setup_mock_city_images(images_dir, "paris")
    tif_path = train_labels_dir / "paris" / "cm" / "paris-cm.tif"
    png_path = train_labels_dir / "paris" / "cm" / "cm.png"
    _create_mock_mask_tiff(tif_path)
    _create_mock_mask_png(png_path)

    # City 2: Only PNG exists -> fallback to PNG
    _setup_mock_city_images(images_dir, "bercy")
    bercy_png = train_labels_dir / "bercy" / "cm" / "cm.png"
    _create_mock_mask_png(bercy_png)

    ds = OSCDDataset(onera_root, split="train", val_cities=(), patch_size=64, stride=64)
    paris_sample = next(s for s in ds.samples if s["city"] == "paris")
    bercy_sample = next(s for s in ds.samples if s["city"] == "bercy")

    # Paris picked the .tif, Bercy picked the .png
    assert paris_sample["label_path"].suffix == ".tif"
    assert paris_sample["label_path"].name == "paris-cm.tif"
    assert bercy_sample["label_path"].suffix == ".png"
    assert bercy_sample["label_path"].name == "cm.png"


# ==============================================================================
# 3. Test missing-label error handling
# ==============================================================================

def test_oscd_missing_label_error_handling(tmp_path: Path) -> None:
    onera_root = tmp_path / "onera DATA"
    images_dir = onera_root / "Onera Satellite Change Detection dataset - Images"

    # Setup city images but DO NOT create any label mask
    _setup_mock_city_images(images_dir, "mumbai")

    # When strict=True, it must raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Ground-truth change mask for OSCD city 'mumbai' not found"):
        OSCDDataset(onera_root, split="train", val_cities=(), patch_size=64, stride=64, strict=True)

    # Direct helper test with required=True
    dummy_ds = object.__new__(OSCDDataset)
    dummy_ds.train_label_dirs = []
    dummy_ds.test_label_dirs = []
    with pytest.raises(FileNotFoundError, match="Ground-truth change mask for OSCD city 'mumbai' not found"):
        OSCDDataset._find_label_file(dummy_ds, images_dir / "mumbai", "mumbai", required=True)


# ==============================================================================
# 4. Test that training and test cities are not mixed
# ==============================================================================

def test_oscd_split_isolation_and_validation(tmp_path: Path) -> None:
    onera_root = tmp_path / "onera DATA"
    images_dir = onera_root / "images"
    train_labels = onera_root / "train_labels"
    test_labels = onera_root / "test_labels"

    # Train cities
    for city in ("paris", "pisa", "mumbai"):
        _setup_mock_city_images(images_dir, city)
        _create_mock_mask_tiff(train_labels / city / "cm" / f"{city}-cm.tif")

    # Test cities
    for city in ("valencia", "dubai"):
        _setup_mock_city_images(images_dir, city)
        _create_mock_mask_tiff(test_labels / city / "cm" / f"{city}-cm.tif")

    # Test that split='train' only contains cities in OSCD_TRAIN_CITIES
    train_ds = OSCDDataset(onera_root, split="train", val_cities=("mumbai",), patch_size=64, stride=64)
    train_cities = {s["city"] for s in train_ds.samples}
    assert train_cities.issubset(set(OSCD_TRAIN_CITIES))
    assert not train_cities.intersection(set(OSCD_TEST_CITIES))
    assert "valencia" not in train_cities
    assert "mumbai" not in train_cities  # excluded because placed in val_cities

    # Test that split='val' only contains specified training cities
    val_ds = OSCDDataset(onera_root, split="val", val_cities=("mumbai",), patch_size=64, stride=64)
    val_cities = {s["city"] for s in val_ds.samples}
    assert val_cities == {"mumbai"}

    # Test that split='test' only contains cities in OSCD_TEST_CITIES
    test_ds = OSCDDataset(onera_root, split="test", patch_size=64, stride=64)
    test_cities = {s["city"] for s in test_ds.samples}
    assert test_cities.issubset(set(OSCD_TEST_CITIES))
    assert not test_cities.intersection(set(OSCD_TRAIN_CITIES))

    # Test that attempting to use a test city as validation raises ValueError
    with pytest.raises(ValueError, match="not in the official OSCD training split"):
        OSCDDataset(onera_root, split="train", val_cities=("valencia",))


# ==============================================================================
# 5. Test image-mask dimension validation (no silent resizing)
# ==============================================================================

def test_oscd_dimension_mismatch_raises_error(tmp_path: Path) -> None:
    onera_root = tmp_path / "onera DATA"
    images_dir = onera_root / "images"
    labels_dir = onera_root / "labels"

    # Images are 64x64, but mask is intentionally mismatched to 32x32
    _setup_mock_city_images(images_dir, "paris", height=64, width=64)
    _create_mock_mask_tiff(labels_dir / "paris" / "cm" / "paris-cm.tif", height=32, width=32)

    with pytest.raises(ValueError, match="OSCD spatial alignment error for city 'paris'"):
        OSCDDataset(onera_root, split="train", val_cities=(), patch_size=32, stride=32)


# ==============================================================================
# 6. Test binary mask conversion {1, 2} -> {0, 1} and {0, 255} -> {0, 1}
# ==============================================================================

def test_oscd_mask_binary_conversion(tmp_path: Path) -> None:
    # A. Official OSCD TIFF with 1=no-change and 2=change
    oscd_tif = tmp_path / "oscd-cm.tif"
    _create_mock_mask_tiff(oscd_tif, height=10, width=10, mode="oscd")
    arr_tif = load_oscd_mask(oscd_tif)
    assert set(np.unique(arr_tif)).issubset({0, 1})
    assert np.all(arr_tif[:5, :] == 1)   # Top half had value 2 -> converted to 1
    assert np.all(arr_tif[5:, :] == 0)   # Bottom half had value 1 -> converted to 0

    # B. PNG with 0=no-change and 255=change
    oscd_png = tmp_path / "cm.png"
    _create_mock_mask_png(oscd_png, height=10, width=10, max_val=255)
    arr_png = load_oscd_mask(oscd_png)
    assert set(np.unique(arr_png)).issubset({0, 1})
    assert np.all(arr_png[:5, :] == 1)   # Top half had value 255 -> converted to 1
    assert np.all(arr_png[5:, :] == 0)   # Bottom half had value 0 -> converted to 0


# ==============================================================================
# 7. Test backward compatibility with legacy unified layout
# ==============================================================================

def test_oscd_legacy_layout_compatibility(tmp_path: Path) -> None:
    legacy_root = tmp_path / "OSCD_legacy"
    city_dir = legacy_root / "dubai"
    _setup_mock_city_images(legacy_root, "dubai")
    _create_mock_mask_png(city_dir / "cm" / "cm.png", height=64, width=64)

    ds = OSCDDataset(legacy_root, split="test", patch_size=64, stride=64)
    assert len(ds) == 1
    item = ds[0]
    assert item["before"].shape == (3, 64, 64)
    assert item["mask"].shape == (64, 64)
    assert set(np.unique(item["mask"].numpy())).issubset({0, 1})
