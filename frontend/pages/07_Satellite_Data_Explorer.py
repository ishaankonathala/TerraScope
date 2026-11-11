"""TerraScope — Satellite Data Explorer & Google Earth Engine Integration.

Search, composite, cloud-mask, preview NDVI, and export aligned Sentinel-2 L2A imagery
via Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED) with graceful offline fallback.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import (
    get_theme_colors,
    render_html,
    render_research_note,
    setup_page,
)
from ml.satellite import NDVI_SCIENTIFIC_NOTE, SENTINEL2_BANDS

setup_page(
    "Satellite Data Explorer (Earth Engine)",
    "Search, cloud-mask, preview multispectral composites, compute NDVI differentials, and export aligned Sentinel-2 L2A observations.",
    icon="🌍",
)

colors = get_theme_colors()
client = get_app_client()
gee_status = client.get_satellite_status()
preset_regions = gee_status.get("preset_regions", {})
is_initialized = bool(gee_status.get("initialized"))

# Connection Status Banner
if is_initialized:
    render_html(
        f"""
        <div style="background-color: {colors['card_bg']}; border-left: 4px solid {colors['forest']}; border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 24px;">
            <div style="font-weight: 700; color: {colors['forest']}; font-size: 0.95rem; margin-bottom: 4px;">
                Google Earth Engine Connected & Ready
            </div>
            <div style="font-size: 0.86rem; color: {colors['muted']};">
                Authenticated via <strong>{gee_status.get('auth_mode', 'standard')}</strong> &bull; GCP Project: <code>{gee_status.get('project_id')}</code> &bull; Catalog: <code>COPERNICUS/S2_SR_HARMONIZED</code>
            </div>
        </div>
        """
    )
else:
    status_code = gee_status.get("status_code", "disabled")
    status_msg = gee_status.get("message", "GEE integration is currently in offline/mock mode.")
    render_html(
        f"""
        <div style="background-color: {colors['card_bg']}; border-left: 4px solid {colors['terracotta']}; border-radius: 0 8px 8px 0; padding: 16px 20px; margin-bottom: 24px;">
            <div style="font-weight: 700; color: {colors['charcoal']}; font-size: 0.95rem; margin-bottom: 4px;">
                Google Earth Engine Status: {status_code.upper()}
            </div>
            <div style="font-size: 0.88rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 8px;">
                {status_msg}
            </div>
            <div style="font-size: 0.82rem; color: {colors['forest']}; font-weight: 600;">
                All local change detection workflows, benchmark datasets, and custom GeoTIFF/RGB uploads remain 100% operational without Earth Engine.
            </div>
        </div>
        """
    )
    with st.expander("How to Authenticate Google Earth Engine", expanded=False):
        st.markdown(
            """
            1. **Google Cloud Project**: Ensure your Google Cloud Project has Earth Engine API enabled at [earthengine.google.com](https://earthengine.google.com/).
            2. **Authentication Options**:
               - **Option A (Local gcloud CLI)**: Run `gcloud auth application-default login` in your terminal.
               - **Option B (Notebook Flow)**: Run `python -c "import ee; ee.Authenticate(auth_mode='notebook')"` to authorize via browser copy-paste code.
               - **Option C (Service Account)**: Place a service account key JSON on disk and set `GEE_SERVICE_ACCOUNT_JSON=/path/to/key.json`.
            3. **Environment**: Set `GEE_ENABLED=true` and `GEE_PROJECT_ID=your-project-id` in `.env`.
            """
        )

# Query & AOI Configuration
st.markdown(
    f"""
    <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
        1. Area of Interest (AOI) & Temporal Windows
    </h3>
    """,
    unsafe_allow_html=True,
)

aoi_mode = st.radio(
    "AOI Selection Method",
    options=["Preset Reference Region", "Center Coordinate + Radius", "Bounding Box Coordinates", "Upload Custom GeoJSON"],
    horizontal=True,
)

bbox_tuple: tuple[float, float, float, float] | None = None
place_name: str | None = None
geojson_geom: dict | None = None

if aoi_mode == "Preset Reference Region":
    region_names = list(preset_regions.keys()) if preset_regions else ["default"]
    place_name = st.selectbox("Select Validated Reference Region", options=region_names)
    if place_name and place_name in preset_regions:
        st.caption(preset_regions[place_name].get("description", ""))
        bbox_tuple = tuple(preset_regions[place_name]["bbox"])  # type: ignore[assignment]
elif aoi_mode == "Center Coordinate + Radius":
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Center Latitude", value=12.8600, min_value=-85.0, max_value=85.0, format="%.4f")
    lon = c2.number_input("Center Longitude", value=77.6900, min_value=-180.0, max_value=180.0, format="%.4f")
    half_deg = c3.slider("Half-Extent (degrees)", min_value=0.01, max_value=0.25, value=0.04, step=0.01)
    bbox_tuple = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
elif aoi_mode == "Bounding Box Coordinates":
    c1, c2, c3, c4 = st.columns(4)
    w = c1.number_input("West Longitude", value=77.6500, format="%.4f")
    s = c2.number_input("South Latitude", value=12.8200, format="%.4f")
    e = c3.number_input("East Longitude", value=77.7300, format="%.4f")
    n = c4.number_input("North Latitude", value=12.9000, format="%.4f")
    bbox_tuple = (float(w), float(s), float(e), float(n))
else:
    gj_file = st.file_uploader("Upload Vector AOI (GeoJSON)", type=["json", "geojson"])
    if gj_file is not None:
        try:
            raw_gj = json.loads(gj_file.getvalue().decode("utf-8"))
            if raw_gj.get("type") == "FeatureCollection" and raw_gj.get("features"):
                geojson_geom = raw_gj["features"][0]["geometry"]
            elif raw_gj.get("type") == "Feature":
                geojson_geom = raw_gj["geometry"]
            else:
                geojson_geom = raw_gj
            st.success("Custom GeoJSON boundary parsed successfully.")
        except Exception as exc:
            st.error(f"Failed to read GeoJSON geometry: {exc}")

d_col1, d_col2, d_col3 = st.columns(3)
today = date.today()
with d_col1:
    b_start = st.date_input("Before Period Start", value=today - timedelta(days=420))
    b_end = st.date_input("Before Period End", value=today - timedelta(days=360))
with d_col2:
    a_start = st.date_input("After Period Start", value=today - timedelta(days=60))
    a_end = st.date_input("After Period End", value=today)
with d_col3:
    max_cloud = st.slider("Max Metadata Cloud Cover (%)", min_value=0, max_value=80, value=20)
    scale_m = st.selectbox("Resolution Target", options=[10, 20, 30, 60], index=1, format_func=lambda s: f"{s} meters / pixel")

b_col1, b_col2 = st.columns([2, 1])
with b_col1:
    selected_bands = st.multiselect(
        "Sentinel-2 Spectral Bands",
        options=[b for b in SENTINEL2_BANDS.keys() if b != "SCL"],
        default=["B4", "B3", "B2", "B8"],
        help="B4=Red, B3=Green, B2=Blue, B8=NIR. Red and NIR are required to compute NDVI.",
    )
with b_col2:
    apply_scl = st.checkbox("Apply Pixel-Level SCL Cloud Mask", value=True, help="Excludes clouds, shadows, and snow using Sentinel-2 Scene Classification Layer.")
    export_dest = st.selectbox("Export Target", options=["local_geotiff", "google_drive", "gcs"], format_func=lambda d: d.replace("_", " ").title())

base_payload = {
    "bbox": list(bbox_tuple) if bbox_tuple else None,
    "place_name": place_name,
    "geojson_geometry": geojson_geom,
    "before_start": b_start.isoformat(),
    "before_end": b_end.isoformat(),
    "after_start": a_start.isoformat(),
    "after_end": a_end.isoformat(),
    "max_cloud_pct": float(max_cloud),
    "bands": selected_bands,
    "scale_m": int(scale_m),
    "target_crs": "EPSG:4326",
    "apply_scl_mask": apply_scl,
    "destination": export_dest,
}

st.write("")
st.markdown(
    f"""
    <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
        2. Query Catalog & Preview Imagery
    </h3>
    """,
    unsafe_allow_html=True,
)

act1, act2, act3 = st.columns(3)
disabled_gee = not is_initialized

with act1:
    if st.button("Search Scene Catalog", use_container_width=True, disabled=disabled_gee):
        with st.spinner("Searching COPERNICUS/S2_SR_HARMONIZED scenes..."):
            try:
                st.session_state["gee_search_res"] = client.search_satellite(base_payload)
            except Exception as exc:
                st.error(f"Search query error: {exc}")

with act2:
    if st.button("Preview Composites & NDVI", use_container_width=True, disabled=disabled_gee):
        with st.spinner("Synthesizing cloud-masked median composites..."):
            try:
                st.session_state["gee_preview_res"] = client.preview_satellite(base_payload)
            except Exception as exc:
                st.error(f"Preview synthesis error: {exc}")

with act3:
    if st.button("Export Aligned GeoTIFFs", type="primary", use_container_width=True, disabled=disabled_gee):
        with st.spinner("Exporting aligned surface reflectance GeoTIFFs..."):
            try:
                exp_res = client.export_satellite(base_payload)
                st.session_state["gee_export_res"] = exp_res
                st.success(f"Export task initialized: {exp_res.get('status')} (Task: `{exp_res.get('task_id')}`)")
            except Exception as exc:
                st.error(f"Export execution failed: {exc}")

# Display Search Results
search_res = st.session_state.get("gee_search_res")
if search_res:
    st.write("")
    render_html(
        f"""
        <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
            Candidate Satellite Scenes
        </h4>
        """
    )
    sc1, sc2 = st.columns(2)
    for col, key, label in ((sc1, "before", "Before Period Scenes"), (sc2, "after", "After Period Scenes")):
        with col:
            part = search_res.get(key, {})
            st.markdown(f"**{label}**")
            if part.get("empty"):
                st.warning(part.get("message", "No scenes match filter parameters."))
            elif part.get("scenes"):
                st.dataframe(pd.DataFrame(part["scenes"]), use_container_width=True, hide_index=True)

# Display Preview Results
preview_res = st.session_state.get("gee_preview_res")
if preview_res:
    st.write("")
    render_html(
        f"""
        <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
            Synthesized Median Surface Composites
        </h4>
        """
    )
    pc1, pc2 = st.columns(2)
    pc1.image(preview_res["before"]["thumb_url"], caption="Before Median Composite (RGB)", use_container_width=True)
    pc2.image(preview_res["after"]["thumb_url"], caption="After Median Composite (RGB)", use_container_width=True)

    ndvi = preview_res.get("ndvi")
    if ndvi and ndvi.get("available"):
        st.write("")
        render_html(
            f"""
            <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
                Normalized Difference Vegetation Index (NDVI)
            </h4>
            """
        )
        nc1, nc2, nc3 = st.columns(3)
        nc1.image(ndvi["before_ndvi_url"], caption="Before NDVI", use_container_width=True)
        nc2.image(ndvi["after_ndvi_url"], caption="After NDVI", use_container_width=True)
        nc3.image(ndvi["diff_ndvi_url"], caption="NDVI Differential (T2 - T1)", use_container_width=True)
        render_research_note(ndvi.get("scientific_note", NDVI_SCIENTIFIC_NOTE))

# Display Export Results & 1-Click Handoff
export_res = st.session_state.get("gee_export_res")
if export_res:
    st.write("")
    render_html(
        f"""
        <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
            Export Pipeline & Model Handoff
        </h4>
        """
    )
    if export_res.get("destination") == "local_geotiff" and export_res.get("before_path"):
        render_html(
            f"""
            <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['forest']}; border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                <div style="font-weight: 600; color: {colors['forest']}; margin-bottom: 4px;">
                    Sentinel-2 Aligned GeoTIFFs Staged on Local Disk
                </div>
                <div style="font-size: 0.85rem; color: {colors['muted']}; margin-bottom: 12px;">
                    Before: <code>{export_res.get('before_path')}</code><br>
                    After: <code>{export_res.get('after_path')}</code>
                </div>
            </div>
            """
        )
        if st.button("Transfer Pair to Change Detection Studio", type="primary"):
            st.session_state["gee_prepared_pair"] = export_res
            st.success("Pair staged! Navigate to 'Change Detection' in the sidebar to execute change detection.")
    elif export_res.get("task_id"):
        st.info(f"Asynchronous Earth Engine Task ID: `{export_res['task_id']}` ({export_res.get('status')})")
        if st.button("Check Task Status"):
            st.session_state["gee_export_res"] = client.get_satellite_task(export_res["task_id"])
            st.rerun()
