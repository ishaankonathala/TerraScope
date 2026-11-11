"""Interactive Folium geospatial map builder for georeferenced change predictions."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import folium
import numpy as np
from PIL import Image
from rasterio.warp import transform_bounds

from ml.preprocessing.image_loader import GeoMetadata


def _rgb_to_data_uri(arr: np.ndarray, transparent_mask: np.ndarray | None = None) -> str:
    if transparent_mask is not None:
        rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
        rgba[:, :, :3] = arr
        rgba[:, :, 3] = np.where(transparent_mask, 200, 0).astype(np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_folium_change_map(
    metadata: GeoMetadata,
    before_rgb: np.ndarray | None = None,
    after_rgb: np.ndarray | None = None,
    binary_mask: np.ndarray | None = None,
    regions_geojson: str | None = None,
    zoom_start: int = 13,
) -> folium.Map | None:
    """Create a Folium web map with satellite imagery and change-mask overlays.

    Returns ``None`` if the imagery lacks valid geographic bounds / CRS.
    """
    if not metadata.is_georeferenced or not metadata.crs or not metadata.bounds:
        return None

    west, south, east, north = metadata.bounds
    try:
        w_lon, s_lat, e_lon, n_lat = transform_bounds(
            metadata.crs, "EPSG:4326", west, south, east, north
        )
    except Exception:
        return None

    center_lat = (s_lat + n_lat) / 2.0
    center_lon = (w_lon + e_lon) / 2.0
    image_bounds = [[s_lat, w_lon], [n_lat, e_lon]]

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite",
        overlay=False,
        control=True,
    ).add_to(fmap)

    if before_rgb is not None:
        folium.raster_layers.ImageOverlay(
            image=_rgb_to_data_uri(before_rgb),
            bounds=image_bounds,
            name="Before (T1)",
            opacity=0.85,
            show=False,
        ).add_to(fmap)

    if after_rgb is not None:
        folium.raster_layers.ImageOverlay(
            image=_rgb_to_data_uri(after_rgb),
            bounds=image_bounds,
            name="After (T2)",
            opacity=0.85,
            show=True,
        ).add_to(fmap)

    if binary_mask is not None:
        mask_rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        mask_rgb[:, :, 0] = 255
        mask_rgb[:, :, 1] = 75
        mask_rgb[:, :, 2] = 45
        folium.raster_layers.ImageOverlay(
            image=_rgb_to_data_uri(mask_rgb, transparent_mask=(binary_mask > 0)),
            bounds=image_bounds,
            name="Predicted Change Mask",
            opacity=0.7,
            show=True,
        ).add_to(fmap)

    if regions_geojson:
        try:
            gj = json.loads(regions_geojson)
            folium.GeoJson(
                gj,
                name="Changed Polygons",
                style_function=lambda _feat: {
                    "color": "#ff4d2e",
                    "weight": 2,
                    "fillColor": "#ff4d2e",
                    "fillOpacity": 0.25,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["region_id", "area_m2", "area_ha"],
                    aliases=["Region ID", "Area (m²)", "Area (ha)"],
                ),
            ).add_to(fmap)
        except Exception:
            pass

    fmap.fit_bounds(image_bounds)
    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap

