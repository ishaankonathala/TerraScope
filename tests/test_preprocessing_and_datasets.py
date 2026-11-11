"""Phase 2 tests for preprocessing pipeline, geospatial alignment, and dataset loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.crs import CRS
from rasterio.transform import from_origin

from ml.datasets import (
    CustomPairDataset,
    LEVIRCDDataset,
    OSCDDataset,
    PairedChangeTransform,
    PairedTransformConfig,
    prepare_custom_dataset,
)
from ml.preprocessing import (
    CloudMaskConfig,
    ImageValidationError,
    align_image_pair,
    apply_cloud_mask,
    extract_patches,
    load_image,
    normalize_pair,
    stitch_tiles,
)


def _save_rgb_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L").save(path)


def _save_geotiff(
    path: Path,
    arr_chw: np.ndarray,
    crs: str,
    transform,
    nodata: float | None = None,
    descriptions: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c, h, w = arr_chw.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=c,
        dtype=arr_chw.dtype,
        crs=CRS.from_user_input(crs),
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(arr_chw)
        if descriptions:
            for i, desc in enumerate(descriptions, start=1):
                dst.set_band_description(i, desc)


def test_image_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ImageValidationError, match="does not exist"):
        load_image(tmp_path / "missing.png")

    bad_ext = tmp_path / "notes.txt"
    bad_ext.write_text("not an image")
    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        load_image(bad_ext)

    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(ImageValidationError, match="empty"):
        load_image(empty_file)

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not-a-valid-png-payload")
    with pytest.raises(ImageValidationError, match="Corrupted"):
        load_image(corrupt)

    tiny = tmp_path / "tiny.png"
    _save_rgb_png(tiny, np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ImageValidationError, match="smaller than minimum"):
        load_image(tiny, min_dim=16)

    valid = tmp_path / "ok.png"
    _save_rgb_png(valid, np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(ImageValidationError, match="exceeds maximum allowed size"):
        load_image(valid, max_size_mb=0.00001)


def test_geotiff_reprojection_alignment(tmp_path: Path) -> None:
    from rasterio.warp import transform_bounds

    # Create a UTM Zone 43N raster (10m pixels) and an overlapping WGS84 EPSG:4326 raster
    utm_tf = from_origin(500000.0, 2100000.0, 10.0, 10.0)
    utm_arr = np.full((3, 64, 64), 120.0, dtype=np.float32)
    t1_path = tmp_path / "t1_utm.tif"
    _save_geotiff(t1_path, utm_arr, "EPSG:32643", utm_tf)

    west, south, east, north = transform_bounds(
        "EPSG:32643", "EPSG:4326", 500000.0, 2100000.0 - 640.0, 500000.0 + 640.0, 2100000.0
    )
    wgs_tf = from_origin(west, north, (east - west) / 48.0, (north - south) / 48.0)
    wgs_arr = np.full((3, 48, 48), 150.0, dtype=np.float32)
    t2_path = tmp_path / "t2_wgs.tif"
    _save_geotiff(t2_path, wgs_arr, "EPSG:4326", wgs_tf)

    before = load_image(t1_path)
    after = load_image(t2_path)
    b_al, a_al, report = align_image_pair(before, after)

    assert report.reprojected is True
    assert report.method == "georeferenced_reprojected"
    assert b_al.metadata.crs == a_al.metadata.crs
    assert b_al.metadata.transform == a_al.metadata.transform
    assert b_al.shape_chw == a_al.shape_chw == (3, 64, 64)


def test_cloud_masking_scl_and_rgb_limitation(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    _save_rgb_png(rgb_path, np.full((32, 32, 3), 100, dtype=np.uint8))
    rgb_img = load_image(rgb_path)
    _, rgb_report = apply_cloud_mask(rgb_img)
    assert rgb_report.cloud_masking_available is False
    assert "unavailable" in rgb_report.limitations_note.lower()

    # Now test Sentinel-2 GeoTIFF with SCL band (class 9 = cloud high probability, 3 = cloud shadow)
    s2_arr = np.full((4, 32, 32), 2000.0, dtype=np.float32)
    scl = np.full((32, 32), 4.0, dtype=np.float32)  # Vegetation
    scl[:8, :8] = 9.0  # Cloud
    scl[8:12, :8] = 3.0  # Shadow
    s2_arr[3] = scl
    s2_path = tmp_path / "s2.tif"
    _save_geotiff(
        s2_path,
        s2_arr,
        "EPSG:32643",
        from_origin(500000.0, 2000000.0, 10.0, 10.0),
        descriptions=["B04", "B03", "B02", "SCL"],
    )
    s2_img = load_image(s2_path)
    masked_img, s2_report = apply_cloud_mask(s2_img, config=CloudMaskConfig(method="scl"))
    assert s2_report.cloud_masking_available is True
    assert s2_report.masked_pixels == (8 * 8 + 4 * 8)
    assert not np.any(masked_img.valid_mask[:12, :8])
    assert np.all(masked_img.valid_mask[12:, :])


def test_tiling_and_smooth_stitching_reconstruction() -> None:
    rng = np.random.default_rng(7)
    arr = rng.random((2, 180, 210), dtype=np.float32)
    patches, specs, orig_hw = extract_patches(arr, tile_size=64, stride=48)
    assert orig_hw == (180, 210)
    assert len(patches) == len(specs)
    reconstructed = stitch_tiles(patches, specs, out_height=180, out_width=210)
    assert reconstructed.shape == arr.shape
    assert np.allclose(arr, reconstructed, atol=1e-5)


def test_paired_augmentations_geometric_consistency() -> None:
    before = np.zeros((3, 64, 64), dtype=np.float32)
    after = np.zeros((3, 64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=np.uint8)

    # Encode a unique asymmetric L-shape target identically in before, after, and mask
    before[:, 10:26, 12:20] = 200.0
    after[:, 10:26, 12:20] = 220.0
    mask[10:26, 12:20] = 1

    cfg = PairedTransformConfig(
        output_size=(64, 64),
        hflip_prob=1.0,
        vflip_prob=1.0,
        rot90_prob=1.0,
        brightness_prob=0.0,
        contrast_prob=0.0,
        gaussian_noise_prob=0.0,
        normalize_mode="unit",
    )
    tf = PairedChangeTransform(config=cfg, is_train=True, seed=99)
    out = tf(before, after, mask=mask)

    b_t = out["before"]
    a_t = out["after"]
    m_t = out["mask"]

    # High-intensity pixels in transformed before & after must align identically with mask == 1
    b_active = b_t[0] > 0.5
    a_active = a_t[0] > 0.5
    m_active = m_t > 0.5
    assert torch.equal(b_active, m_active)
    assert torch.equal(a_active, m_active)
    assert int(m_active.sum()) == 16 * 8


def test_levir_cd_loader_and_patching(tmp_path: Path) -> None:
    root = tmp_path / "LEVIR-CD"
    for split in ("train", "val", "test"):
        img_a = np.full((128, 128, 3), 50, dtype=np.uint8)
        img_b = np.full((128, 128, 3), 60, dtype=np.uint8)
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[16:48, 16:48] = 1
        _save_rgb_png(root / split / "A" / "sample_1.png", img_a)
        _save_rgb_png(root / split / "B" / "sample_1.png", img_b)
        _save_mask_png(root / split / "label" / "sample_1.png", mask)

    ds = LEVIRCDDataset(root, split="train", patch_size=64, stride=64)
    assert len(ds) == 4  # 128x128 tiled into four 64x64 patches
    sample = ds[0]
    assert sample["before"].shape == (3, 64, 64)
    assert sample["after"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)
    assert sample["meta"]["dataset"] == "LEVIR-CD"


def test_oscd_multispectral_loader_resampling(tmp_path: Path) -> None:
    oscd_root = tmp_path / "OSCD"
    city_dir = oscd_root / "images" / "dubai"  # in OSCD_TEST_CITIES
    tf_10m = from_origin(300000.0, 2780000.0, 10.0, 10.0)
    tf_20m = from_origin(300000.0, 2780000.0, 20.0, 20.0)

    for folder in ("imgs_1_rect", "imgs_2_rect"):
        _save_geotiff(
            city_dir / folder / "B04.tif",
            np.full((1, 64, 64), 1200.0, dtype=np.float32),
            "EPSG:32640",
            tf_10m,
        )
        _save_geotiff(
            city_dir / folder / "B08.tif",
            np.full((1, 64, 64), 2500.0, dtype=np.float32),
            "EPSG:32640",
            tf_10m,
        )
        # B11 is natively 20m (32x32) and must be resampled cleanly to 10m (64x64)
        _save_geotiff(
            city_dir / folder / "B11.tif",
            np.full((1, 32, 32), 1800.0, dtype=np.float32),
            "EPSG:32640",
            tf_20m,
        )

    _save_mask_png(city_dir / "cm" / "cm.png", np.ones((64, 64), dtype=np.uint8))

    ds = OSCDDataset(
        oscd_root,
        split="test",
        bands=("B04", "B08", "B11"),
        patch_size=64,
        stride=64,
    )
    assert len(ds) == 1
    item = ds[0]
    assert item["before"].shape == (3, 64, 64)
    assert item["meta"]["geospatial"]["crs"] == "EPSG:32640"


def test_custom_dataset_and_prepare_utility(tmp_path: Path) -> None:
    raw_dir = tmp_path / "custom_raw"
    prep_dir = tmp_path / "custom_prepared"

    _save_rgb_png(raw_dir / "before" / "site1.png", np.full((96, 96, 3), 80, dtype=np.uint8))
    _save_rgb_png(raw_dir / "after" / "site1.png", np.full((96, 96, 3), 120, dtype=np.uint8))
    _save_mask_png(raw_dir / "mask" / "site1.png", np.ones((96, 96), dtype=np.uint8))

    summary = prepare_custom_dataset(raw_dir, prep_dir, patch_size=64, stride=32)
    assert summary["source_pairs"] == 1
    assert summary["generated_patches"] == 4
    assert (prep_dir / "manifest.json").is_file()

    ds = CustomPairDataset(root_dir=prep_dir, image_size=(64, 64))
    assert len(ds) == 4
    sample = ds[0]
    assert sample["before"].shape == (3, 64, 64)
    assert sample["mask"].shape == (64, 64)


def test_normalize_pair_preserves_temporal_delta() -> None:
    b = np.full((3, 16, 16), 100.0, dtype=np.float32)
    a = np.full((3, 16, 16), 200.0, dtype=np.float32)
    b_n, a_n = normalize_pair(b, a, mode="zscore")
    assert np.mean(a_n) > np.mean(b_n)
