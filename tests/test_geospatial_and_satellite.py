"""Phase 10-11 tests for geospatial area analysis, Folium maps, NDVI, and GEE client."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from ml.geospatial import (
    build_folium_change_map,
    compute_geospatial_change_stats,
    save_georeferenced_raster,
)
from ml.preprocessing import GeoMetadata, LoadedImage
from ml.satellite import (
    AOISpec,
    GEEConfig,
    compare_ndvi,
    compute_ndvi,
    compute_ndvi_from_loaded_pair,
    get_gee_status,
    search_sentinel2_scenes,
    validate_sentinel2_params,
)


def test_projected_and_geodesic_area_consistency(tmp_path: Path) -> None:
    # 100x100 raster at 10m pixel spacing -> 1 pixel = 100 m^2
    tf_utm = from_origin(500000.0, 2100000.0, 10.0, 10.0)
    meta_utm = GeoMetadata(
        width=100,
        height=100,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=tuple(float(x) for x in tf_utm[:6]),
        bounds=(500000.0, 2099000.0, 501000.0, 2100000.0),
        is_georeferenced=True,
    )
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:40, 20:40] = 1  # 400 changed pixels -> 40,000 m^2 = 4.0 ha
    valid = np.ones((100, 100), dtype=bool)

    stats_utm = compute_geospatial_change_stats(mask, valid, meta_utm)
    assert stats_utm["is_georeferenced"] is True
    assert pytest.approx(stats_utm["changed_area_m2"], rel=1e-4) == 40000.0
    assert pytest.approx(stats_utm["changed_area_ha"], rel=1e-4) == 4.0
    assert stats_utm["regions_geojson"] is not None

    # Equivalent raster in geographic EPSG:4326 degrees
    w, s, e, n = transform_bounds("EPSG:32643", "EPSG:4326", 500000.0, 2099000.0, 501000.0, 2100000.0)
    tf_deg = from_origin(w, n, (e - w) / 100.0, (n - s) / 100.0)
    meta_deg = GeoMetadata(
        width=100,
        height=100,
        count=3,
        dtype="uint8",
        crs="EPSG:4326",
        transform=tuple(float(x) for x in tf_deg[:6]),
        bounds=(w, s, e, n),
        is_georeferenced=True,
    )
    stats_deg = compute_geospatial_change_stats(mask, valid, meta_deg)
    assert "geodesic_wgs84" in stats_deg["area_calculation_method"]
    # Geodesic area of same footprint must match UTM area within ~1%
    assert abs(stats_deg["changed_area_m2"] - 40000.0) / 40000.0 < 0.01

    # Save GeoTIFF and verify CRS/transform preserved
    tif_out = save_georeferenced_raster(mask, meta_utm, tmp_path / "change.tif", nodata=255)
    with rasterio.open(tif_out) as src:
        assert src.crs.to_string() == "EPSG:32643"
        assert tuple(src.transform[:6]) == meta_utm.transform

    # Build interactive Folium map
    rgb = np.full((100, 100, 3), 110, dtype=np.uint8)
    fmap = build_folium_change_map(
        meta_utm,
        before_rgb=rgb,
        after_rgb=rgb,
        binary_mask=mask,
        regions_geojson=stats_utm["regions_geojson"],
    )
    assert fmap is not None
    assert "leaflet" in fmap.get_root().render().lower()


def test_non_georeferenced_stats_do_not_invent_area() -> None:
    meta = GeoMetadata(width=64, height=64, count=3, dtype="uint8", is_georeferenced=False)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:32, :32] = 1
    stats = compute_geospatial_change_stats(mask, np.ones((64, 64), dtype=bool), meta)
    assert stats["is_georeferenced"] is False
    assert stats["changed_area_m2"] is None
    assert stats["area_calculation_method"] == "image_relative_only"
    assert pytest.approx(stats["change_percentage"], rel=1e-4) == 25.0


def test_ndvi_calculation_zero_guard_and_loaded_pair() -> None:
    red = np.array([[1000.0, 0.0], [2000.0, 500.0]], dtype=np.float32)
    nir = np.array([[3000.0, 0.0], [2000.0, 1500.0]], dtype=np.float32)
    valid = np.array([[True, True], [True, False]])

    ndvi, stats = compute_ndvi(red, nir, valid_mask=valid)
    assert pytest.approx(ndvi[0, 0], rel=1e-4) == 0.5  # (3000-1000)/4000
    assert np.isnan(ndvi[0, 1])  # zero denominator guarded
    assert pytest.approx(ndvi[1, 0], abs=1e-5) == 0.0
    assert np.isnan(ndvi[1, 1])  # masked pixel excluded
    assert stats["valid_pixels"] == 2

    comp = compare_ndvi(red, nir, red * 1.2, nir * 0.8, before_valid=valid, after_valid=valid)
    assert comp["available"] is True
    assert comp["diff_stats"]["mean_diff"] < 0.0

    # LoadedImage helper
    meta = GeoMetadata(width=2, height=2, count=2, dtype="float32", band_names=["B4", "B8"])
    b_img = LoadedImage(data=np.stack([red, nir]), valid_mask=valid, metadata=meta, raw_dtype="float32")
    a_img = LoadedImage(
        data=np.stack([red * 1.1, nir * 0.9]), valid_mask=valid, metadata=meta, raw_dtype="float32"
    )
    pair_ndvi = compute_ndvi_from_loaded_pair(b_img, a_img)
    assert pair_ndvi["available"] is True


def test_gee_client_graceful_status_and_validation() -> None:
    # When GEE is disabled (default), status is cleanly 'disabled' without errors
    st = get_gee_status(GEEConfig(enabled=False))
    assert st.status_code == "disabled"
    assert st.initialized is False

    # Missing project ID when enabled without service account
    st_no_proj = get_gee_status(GEEConfig(enabled=True, project_id=None))
    assert st_no_proj.status_code == "missing_project"

    # Parameter validation guards
    bands, scale = validate_sentinel2_params(["B4", "b3", "B2"], "2024-01-01", "2024-02-01", 20.0, 10)
    assert bands == ["B4", "B3", "B2"]
    assert scale == 10

    with pytest.raises(ValueError, match="Unsupported Sentinel-2 band"):
        validate_sentinel2_params(["INVALID"], "2024-01-01", "2024-02-01", 20.0, 10)

    with pytest.raises(ValueError, match="end_date"):
        validate_sentinel2_params(["B4"], "2024-03-01", "2024-02-01", 20.0, 10)

    aoi = AOISpec(place_name="Bengaluru Urban Expansion Corridor")
    bounds = aoi.resolve_bounds()
    assert len(bounds) == 4

    with pytest.raises(ValueError, match="exceeds 2.0 degrees"):
        AOISpec(bbox=(70.0, 10.0, 75.0, 15.0)).resolve_bounds()


def test_gee_search_mocked_empty_and_valid_scenes() -> None:
    aoi = AOISpec(bbox=(77.65, 12.82, 77.70, 12.87))

    mock_ee = MagicMock()
    mock_col = MagicMock()
    mock_ee.ImageCollection.return_value = mock_col
    mock_col.filterBounds.return_value = mock_col
    mock_col.filterDate.return_value = mock_col
    mock_col.filter.return_value = mock_col
    mock_col.sort.return_value = mock_col
    mock_col.limit.return_value = mock_col
    mock_col.getInfo.return_value = {
        "features": [
            {
                "id": "COPERNICUS/S2_SR_HARMONIZED/20240115T050151_T43PGQ",
                "properties": {
                    "system:time_start": 1705294911000,
                    "CLOUDY_PIXEL_PERCENTAGE": 4.2,
                    "MGRS_TILE": "43PGQ",
                    "SPACECRAFT_NAME": "Sentinel-2B",
                },
            }
        ]
    }

    ready_status = MagicMock(initialized=True, status_code="ready", message="ok")
    with (
        patch("ml.satellite.sentinel2.initialize_gee", return_value=ready_status),
        patch("ml.satellite.sentinel2.ee", mock_ee),
    ):
        res = search_sentinel2_scenes(aoi, "2024-01-01", "2024-02-01", max_cloud_pct=20.0)
        assert res["ok"] is True
        assert res["empty"] is False
        assert len(res["scenes"]) == 1
        assert res["scenes"][0]["mgrs_tile"] == "43PGQ"

        mock_col.getInfo.return_value = {"features": []}
        empty_res = search_sentinel2_scenes(aoi, "2024-01-01", "2024-02-01", max_cloud_pct=5.0)
        assert empty_res["ok"] is True
        assert empty_res["empty"] is True

