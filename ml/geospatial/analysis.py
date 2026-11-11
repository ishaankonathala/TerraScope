"""Geospatial change-area analytics, GeoTIFF preservation, and GeoJSON vectorization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyproj
import rasterio
from affine import Affine
from pyproj import CRS as PyprojCRS
from pyproj import Geod
from rasterio.features import shapes
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from ml.preprocessing.image_loader import GeoMetadata


def compute_pixel_areas_m2(metadata: GeoMetadata) -> tuple[np.ndarray | None, str]:
    """Compute per-pixel area in square meters when valid georeferencing exists.

    - For projected CRS rasters, computes pixel area from the affine determinant
      and linear unit conversion factor to meters.
    - For geographic (lat/lon degree) CRS rasters, computes per-row WGS84 geodesic
      cell areas via ``pyproj.Geod`` instead of treating degrees as meters.
    - For non-georeferenced images (PNG/JPG), returns ``(None, "image_relative_only")``.
    """
    if not metadata.is_georeferenced or metadata.crs is None or metadata.transform is None:
        return None, "image_relative_only"

    crs = PyprojCRS.from_user_input(metadata.crs)
    tf = Affine(*metadata.transform)
    h, w = metadata.height, metadata.width

    if crs.is_projected:
        unit_factor = float(crs.axis_info[0].unit_conversion_factor or 1.0)
        pixel_area = abs(tf.a * tf.e - tf.b * tf.d) * (unit_factor**2)
        return np.full((h, w), pixel_area, dtype=np.float64), f"projected_crs ({metadata.crs})"

    if crs.is_geographic:
        geod = Geod(ellps="WGS84")
        row_areas = np.zeros(h, dtype=np.float64)
        # For each row, compute exact geodesic area of a single pixel polygon at column 0
        for r in range(h):
            lon0, lat0 = tf @ (0.0, float(r))
            lon1, lat1 = tf @ (1.0, float(r))
            lon2, lat2 = tf @ (1.0, float(r + 1))
            lon3, lat3 = tf @ (0.0, float(r + 1))
            area, _ = geod.polygon_area_perimeter(
                [lon0, lon1, lon2, lon3], [lat0, lat1, lat2, lat3]
            )
            row_areas[r] = abs(float(area))
        area_grid = np.repeat(row_areas[:, None], w, axis=1)
        return area_grid, f"geodesic_wgs84 ({metadata.crs})"

    return None, "unsupported_crs"


def extract_changed_polygons_gdf(
    binary_mask: np.ndarray,
    metadata: GeoMetadata,
    valid_mask: np.ndarray | None = None,
    max_polygons: int = 500,
) -> gpd.GeoDataFrame | None:
    """Vectorize connected binary change regions into a GeoDataFrame in EPSG:4326."""
    if not metadata.is_georeferenced or metadata.crs is None or metadata.transform is None:
        return None

    tf = Affine(*metadata.transform)
    mask_u8 = (binary_mask > 0).astype(np.uint8)
    if valid_mask is not None:
        mask_u8 = mask_u8 & valid_mask.astype(np.uint8)

    geoms = []
    for geom_dict, val in shapes(mask_u8, mask=(mask_u8 == 1), transform=tf):
        if int(val) == 1:
            geoms.append(shape(geom_dict))
            if len(geoms) >= max_polygons:
                break

    if not geoms:
        return gpd.GeoDataFrame(
            {"region_id": [], "area_m2": [], "area_ha": []},
            geometry=[],
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame({"region_id": list(range(1, len(geoms) + 1))}, geometry=geoms, crs=metadata.crs)
    # Compute accurate area using cylindrical equal-area projection (EPSG:6933)
    try:
        gdf_eq = gdf.to_crs("EPSG:6933")
        areas_m2 = gdf_eq.geometry.area.to_numpy(dtype=np.float64)
    except Exception:
        areas_m2 = np.zeros(len(geoms), dtype=np.float64)

    gdf["area_m2"] = np.round(areas_m2, 2)
    gdf["area_ha"] = np.round(areas_m2 / 10000.0, 4)
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    gdf_wgs84 = gdf_wgs84.sort_values(by="area_m2", ascending=False).reset_index(drop=True)
    return gdf_wgs84


def compute_geospatial_change_stats(
    binary_mask: np.ndarray,
    valid_mask: np.ndarray,
    metadata: GeoMetadata,
) -> dict[str, Any]:
    """Compute image-relative and (when georeferenced) real-world area change statistics."""
    valid_b = valid_mask.astype(bool)
    changed_b = (binary_mask > 0) & valid_b

    changed_px = int(np.sum(changed_b))
    valid_px = int(np.sum(valid_b))
    total_px = int(binary_mask.shape[0] * binary_mask.shape[1])
    change_pct = (changed_px / valid_px * 100.0) if valid_px > 0 else 0.0

    area_grid, area_method = compute_pixel_areas_m2(metadata)
    if area_grid is None:
        return {
            "is_georeferenced": False,
            "crs": None,
            "bounds_wgs84": None,
            "center_latlon": None,
            "area_calculation_method": area_method,
            "changed_pixels": changed_px,
            "valid_pixels": valid_px,
            "total_pixels": total_px,
            "change_percentage": round(change_pct, 4),
            "changed_area_m2": None,
            "changed_area_ha": None,
            "changed_area_km2": None,
            "valid_area_m2": None,
            "valid_area_km2": None,
            "note": (
                "Input imagery has no CRS metadata; displaying image-relative pixel percentages "
                "rather than real-world geographic area."
            ),
            "regions_geojson": None,
            "top_regions": [],
        }

    changed_m2 = float(np.sum(area_grid[changed_b]))
    valid_m2 = float(np.sum(area_grid[valid_b]))
    west, south, east, north = metadata.bounds or rasterio.transform.array_bounds(
        metadata.height, metadata.width, Affine(*metadata.transform)
    )
    try:
        wgs_bounds = transform_bounds(metadata.crs, "EPSG:4326", west, south, east, north)
        center_latlon = [
            round((wgs_bounds[1] + wgs_bounds[3]) / 2.0, 6),
            round((wgs_bounds[0] + wgs_bounds[2]) / 2.0, 6),
        ]
    except Exception:
        wgs_bounds = None
        center_latlon = None

    gdf = extract_changed_polygons_gdf(binary_mask, metadata, valid_mask=valid_b)
    geojson_str = gdf.to_json() if gdf is not None and len(gdf) > 0 else None
    top_regions = (
        gdf[["region_id", "area_m2", "area_ha"]].head(15).to_dict(orient="records")
        if gdf is not None and len(gdf) > 0
        else []
    )

    return {
        "is_georeferenced": True,
        "crs": metadata.crs,
        "bounds_wgs84": [round(float(x), 6) for x in wgs_bounds] if wgs_bounds else None,
        "center_latlon": center_latlon,
        "area_calculation_method": area_method,
        "changed_pixels": changed_px,
        "valid_pixels": valid_px,
        "total_pixels": total_px,
        "change_percentage": round(change_pct, 4),
        "changed_area_m2": round(changed_m2, 2),
        "changed_area_ha": round(changed_m2 / 10000.0, 4),
        "changed_area_km2": round(changed_m2 / 1_000_000.0, 6),
        "valid_area_m2": round(valid_m2, 2),
        "valid_area_km2": round(valid_m2 / 1_000_000.0, 6),
        "note": f"Computed real-world area via {area_method} over valid pixels only.",
        "regions_geojson": geojson_str,
        "top_regions": top_regions,
    }


def save_georeferenced_raster(
    array: np.ndarray,
    metadata: GeoMetadata,
    output_path: str | Path,
    nodata: float | None = None,
) -> Path:
    """Save a prediction mask or probability raster preserving CRS and affine transform."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = array[None, :, :] if array.ndim == 2 else array
    c, h, w = arr.shape

    if metadata.is_georeferenced and metadata.crs and metadata.transform:
        with rasterio.open(
            out,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=c,
            dtype=arr.dtype,
            crs=rasterio.crs.CRS.from_user_input(metadata.crs),
            transform=Affine(*metadata.transform),
            nodata=nodata,
            compress="deflate",
        ) as dst:
            dst.write(arr)
    else:
        from PIL import Image

        if arr.dtype == np.float32 or arr.dtype == np.float64:
            vis = np.clip(np.round(arr[0] * 255.0), 0, 255).astype(np.uint8)
        else:
            vis = (arr[0] > 0).astype(np.uint8) * 255
        Image.fromarray(vis, mode="L").save(out)
    return out
