"""Geospatial Analysis page: CRS metadata, real-world area statistics, and Folium maps."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import render_metric_cards, setup_page
from ml.geospatial import build_folium_change_map
from ml.preprocessing.image_loader import GeoMetadata

setup_page(
    "Geospatial Analysis & Area Estimation",
    "Inspect CRS projections, geodesic/equal-area affected areas, region polygons, and interactive maps.",
    icon="🗺️",
)

client = get_app_client()
results = client.list_results(limit=25)

if not results:
    st.info(
        "No analyzed results available yet. Run inference on a GeoTIFF or PNG pair in "
        "**Change Detection** first."
    )
    st.stop()

options = [r["result_id"] for r in results]
default_idx = 0
if st.session_state.get("last_result_id") in options:
    default_idx = options.index(st.session_state["last_result_id"])

rid = st.selectbox("Select Analysis Result", options=options, index=default_idx)
res = client.get_result(rid)
geo = res.get("geospatial", {})

if not geo.get("is_georeferenced"):
    st.warning(geo.get("note", "Image is not georeferenced; displaying image-relative statistics only."))
    render_metric_cards(
        [
            {"title": "CRS", "value": "Not Georeferenced"},
            {"title": "Changed Pixels", "value": f"{res['changed_pixels']:,}"},
            {"title": "Change Share", "value": f"{res['change_percentage']:.2f}%"},
            {"title": "Area Method", "value": "Image-Relative"},
        ]
    )
else:
    render_metric_cards(
        [
            {"title": "CRS", "value": str(geo.get("crs"))},
            {
                "title": "Changed Area",
                "value": f"{geo['changed_area_ha']:.4f} ha",
                "sub": f"{geo['changed_area_m2']:,.1f} m² ({geo['changed_area_km2']:.6f} km²)",
            },
            {
                "title": "Valid Footprint",
                "value": f"{geo['valid_area_km2']:.4f} km²",
                "sub": geo.get("area_calculation_method", ""),
            },
            {"title": "Change Share", "value": f"{res['change_percentage']:.2f}%"},
        ]
    )

    b_meta_dict = res.get("metadata", {}).get("before_metadata", {})
    meta = GeoMetadata(
        width=int(b_meta_dict.get("width", 0)),
        height=int(b_meta_dict.get("height", 0)),
        count=int(b_meta_dict.get("count", 3)),
        dtype=str(b_meta_dict.get("dtype", "float32")),
        crs=b_meta_dict.get("crs"),
        transform=tuple(b_meta_dict["transform"]) if b_meta_dict.get("transform") else None,
        bounds=tuple(b_meta_dict["bounds"]) if b_meta_dict.get("bounds") else None,
        is_georeferenced=bool(b_meta_dict.get("is_georeferenced")),
    )

    def _bytes_to_rgb(data: bytes | None) -> np.ndarray | None:
        if not data:
            return None
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    b_rgb = _bytes_to_rgb(client.get_artifact_bytes(rid, "before_png"))
    a_rgb = _bytes_to_rgb(client.get_artifact_bytes(rid, "after_png"))
    m_bytes = client.get_artifact_bytes(rid, "png")
    m_arr = (
        (np.asarray(Image.open(io.BytesIO(m_bytes)).convert("L")) > 0).astype(np.uint8)
        if m_bytes
        else None
    )

    fmap = build_folium_change_map(
        metadata=meta,
        before_rgb=b_rgb,
        after_rgb=a_rgb,
        binary_mask=m_arr,
        regions_geojson=geo.get("regions_geojson"),
    )
    if fmap is not None:
        st.markdown("### Interactive Georeferenced Map")
        components.html(fmap.get_root().render(), height=540, scrolling=False)

    top_regions = geo.get("top_regions", [])
    if top_regions:
        st.markdown("### Region-Level Changed Area Breakdown")
        st.dataframe(pd.DataFrame(top_regions), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        tif_b = client.get_artifact_bytes(rid, "geotiff")
        if tif_b:
            st.download_button(
                "Download Georeferenced Change Mask (GeoTIFF)",
                data=tif_b,
                file_name=f"{rid}_change_mask.tif",
                mime="image/tiff",
                use_container_width=True,
            )
    with c2:
        gj_b = client.get_artifact_bytes(rid, "geojson")
        if gj_b:
            st.download_button(
                "Download Changed Polygons (GeoJSON)",
                data=gj_b,
                file_name=f"{rid}_regions.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

