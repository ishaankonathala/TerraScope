"""TerraScope — Geospatial Analysis, CRS Projections & Area Accounting.

Distinguishes rigorous georeferenced ground metrics (hectares, m², km², EPSG projections)
from image-relative pixel statistics, and embeds interactive GIS map views.
"""

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
from frontend.components.theme import (
    get_theme_colors,
    render_metric_cards,
    render_research_note,
    setup_page,
)
from ml.geospatial import build_folium_change_map
from ml.preprocessing.image_loader import GeoMetadata

setup_page(
    "Geospatial Intelligence & Area Accounting",
    "Inspect projection references, geodesic surface measurements, vector polygon geometries, and interactive map layers.",
    icon="🗺️",
)

colors = get_theme_colors()
client = get_app_client()
results = client.list_results(limit=30)

if not results:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px dashed {colors['border']}; border-radius: 8px; padding: 32px; text-align: center;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 6px;">
                No Prior Analysis Records Found
            </div>
            <p style="color: {colors['muted']}; font-size: 0.92rem; margin: 0;">
                Run an inference on a GeoTIFF or satellite image pair in the <strong>Change Detection</strong> studio first.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

options = [r["result_id"] for r in results]
default_idx = 0
if st.session_state.get("last_result_id") in options:
    default_idx = options.index(st.session_state["last_result_id"])

col_sel, col_stat = st.columns([2, 1])
with col_sel:
    rid = st.selectbox("Select Analysis Record", options=options, index=default_idx)
    res = client.get_result(rid)
    geo = res.get("geospatial", {})

with col_stat:
    is_geo = bool(geo.get("is_georeferenced"))
    status_color = colors["forest"] if is_geo else colors["terracotta"]
    status_label = "Georeferenced (Real-World Area Valid)" if is_geo else "Image-Relative (Unprojected)"
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 14px 18px; margin-top: 24px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['sage']}; font-weight: 700; margin-bottom: 4px;">
                Spatial Coordinate State
            </div>
            <div style="font-size: 0.9rem; font-weight: 700; color: {status_color};">
                {status_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

if not is_geo:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border-left: 4px solid {colors['terracotta']}; padding: 14px 18px; border-radius: 0 8px 8px 0; margin-bottom: 20px;">
            <div style="font-weight: 600; color: {colors['charcoal']}; font-size: 0.92rem; margin-bottom: 4px;">
                Image-Relative Mode — Spatial Coordinates Unavailable
            </div>
            <div style="font-size: 0.86rem; color: {colors['muted']}; line-height: 1.5;">
                This analysis was performed on standard non-georeferenced images (e.g. PNG/JPEG). Physical metric units (such as hectares, square meters, and geodetic polygons) cannot be honestly computed without coordinate reference system (CRS) headers and an affine geotransform matrix. Only pixel-relative proportions are reported below.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_metric_cards(
        [
            {"title": "Coordinate Reference", "value": "None (Pixel Grid)"},
            {"title": "Changed Pixels", "value": f"{res.get('changed_pixels', 0):,}"},
            {"title": "Change Ratio", "value": f"{res.get('change_percentage', 0.0):.2f}%"},
            {"title": "Total Valid Footprint", "value": f"{res.get('total_valid_pixels', 0):,} px"},
        ]
    )
else:
    render_metric_cards(
        [
            {"title": "Projection (CRS)", "value": str(geo.get("crs", "Unknown"))},
            {
                "title": "Changed Area",
                "value": f"{geo.get('changed_area_ha', 0.0):.4f} ha",
                "sub": f"{geo.get('changed_area_m2', 0.0):,.1f} m² ({geo.get('changed_area_km2', 0.0):.6f} km²)",
            },
            {
                "title": "Total Footprint",
                "value": f"{geo.get('valid_area_km2', 0.0):.4f} km²",
                "sub": f"Calculation: {geo.get('area_calculation_method', 'Projected UTM')}",
            },
            {"title": "Change Percentage", "value": f"{res.get('change_percentage', 0.0):.2f}%"},
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
        st.write("")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
                <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {colors['forest']}; margin: 0;">
                    Interactive Georeferenced Map Inspection
                </h4>
                <span style="font-size: 0.82rem; color: {colors['muted']};">True spatial bounds &bull; OpenStreetMap & Esri World Imagery</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        components.html(fmap.get_root().render(), height=520, scrolling=False)

    top_regions = geo.get("top_regions", [])
    if top_regions:
        st.write("")
        st.markdown(
            f"""
            <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
                Polygon-Level Change Geometry Breakdown
            </h4>
            <p style="font-size: 0.88rem; color: {colors['muted']}; margin-bottom: 12px;">
                Detected change clusters vectorized into geodetic polygon boundaries with centroid coordinates and physical area calculation.
            </p>
            """,
            unsafe_allow_html=True,
        )
        reg_df = pd.DataFrame(top_regions)
        if "area_ha" in reg_df.columns:
            reg_df["area_ha"] = reg_df["area_ha"].round(4)
        if "area_m2" in reg_df.columns:
            reg_df["area_m2"] = reg_df["area_m2"].round(1)
        st.dataframe(reg_df, use_container_width=True, hide_index=True)

st.write("")
render_research_note(
    "Scientific Footprint Note: Area measurements are projected using local equal-area or UTM projections rather than planar degrees to prevent latitude-dependent distortions. Causal labeling (deforestation vs. construction) requires secondary domain-specific multispectral classifiers."
)
st.write("")

# Geospatial Exports
st.markdown(
    f"""
    <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 12px;">
        Export GIS Formats
    </h4>
    """,
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)
with c1:
    tif_b = client.get_artifact_bytes(rid, "geotiff")
    if tif_b:
        st.download_button(
            "Download Georeferenced Change Mask (.tif)",
            data=tif_b,
            file_name=f"{rid}_change_mask.tif",
            mime="image/tiff",
            use_container_width=True,
        )
    else:
        st.button("GeoTIFF Mask Unavailable (Unprojected)", disabled=True, use_container_width=True)

with c2:
    gj_b = client.get_artifact_bytes(rid, "geojson")
    if gj_b:
        st.download_button(
            "Download Vector Polygons (.geojson)",
            data=gj_b,
            file_name=f"{rid}_regions.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
    else:
        st.button("GeoJSON Polygons Unavailable", disabled=True, use_container_width=True)
