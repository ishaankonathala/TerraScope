"""Satellite Data Explorer powered by Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED)."""

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
from frontend.components.theme import setup_page
from ml.satellite import NDVI_SCIENTIFIC_NOTE, SENTINEL2_BANDS

setup_page(
    "Satellite Data Explorer (Google Earth Engine)",
    "Search, mask, preview, compute NDVI, and export aligned Sentinel-2 L2A imagery via Google Earth Engine.",
    icon="🌍",
)

client = get_app_client()
gee_status = client.get_satellite_status()
preset_regions = gee_status.get("preset_regions", {})

if gee_status.get("initialized"):
    st.success(f"GEE Ready ({gee_status.get('auth_mode')}, project `{gee_status.get('project_id')}`).")
else:
    st.warning(
        f"**GEE Status: `{gee_status.get('status_code')}`** — {gee_status.get('message')}\n\n"
        "Local uploads and LEVIR-CD / OSCD workflows remain fully functional without GEE."
    )
    with st.expander("How to Enable Google Earth Engine", expanded=False):
        st.markdown(
            """
            1. Register a Google Cloud project with Earth Engine enabled (`https://earthengine.google.com/`).
            2. Run `earthengine authenticate` locally (or configure a service account key).
            3. Set `GEE_ENABLED=true` and `GEE_PROJECT_ID=<your-project>` in `.env`.
            4. Restart the FastAPI/Streamlit app. See `docs/gee_documentation.md` for quota and academic eligibility terms.
            """
        )

st.markdown("### 1. Area of Interest & Observation Windows")
aoi_mode = st.radio(
    "AOI Input Mode",
    options=["Preset Region", "Lat/Lon Center + Radius", "Bounding Box (W, S, E, N)", "Upload GeoJSON"],
    horizontal=True,
)

bbox_tuple: tuple[float, float, float, float] | None = None
place_name: str | None = None
geojson_geom: dict | None = None

if aoi_mode == "Preset Region":
    region_names = list(preset_regions.keys())
    place_name = st.selectbox("Select Reference Region", options=region_names)
    if place_name:
        st.caption(preset_regions[place_name]["description"])
        bbox_tuple = tuple(preset_regions[place_name]["bbox"])  # type: ignore[assignment]
elif aoi_mode == "Lat/Lon Center + Radius":
    c1, c2, c3 = st.columns(3)
    lat = c1.number_input("Center Latitude", value=12.86, min_value=-85.0, max_value=85.0, format="%.4f")
    lon = c2.number_input("Center Longitude", value=77.69, min_value=-180.0, max_value=180.0, format="%.4f")
    half_deg = c3.slider("Half-Extent (deg)", min_value=0.01, max_value=0.25, value=0.04, step=0.01)
    bbox_tuple = (lon - half_deg, lat - half_deg, lon + half_deg, lat + half_deg)
elif aoi_mode == "Bounding Box (W, S, E, N)":
    c1, c2, c3, c4 = st.columns(4)
    w = c1.number_input("West Lon", value=77.65, format="%.4f")
    s = c2.number_input("South Lat", value=12.82, format="%.4f")
    e = c3.number_input("East Lon", value=77.73, format="%.4f")
    n = c4.number_input("North Lat", value=12.90, format="%.4f")
    bbox_tuple = (float(w), float(s), float(e), float(n))
else:
    gj_file = st.file_uploader("Upload GeoJSON AOI Polygon", type=["json", "geojson"])
    if gj_file is not None:
        raw_gj = json.loads(gj_file.getvalue().decode("utf-8"))
        if raw_gj.get("type") == "FeatureCollection":
            geojson_geom = raw_gj["features"][0]["geometry"]
        elif raw_gj.get("type") == "Feature":
            geojson_geom = raw_gj["geometry"]
        else:
            geojson_geom = raw_gj

d_col1, d_col2, d_col3 = st.columns(3)
with d_col1:
    today = date.today()
    b_start = st.date_input("Before Start", value=today - timedelta(days=420))
    b_end = st.date_input("Before End", value=today - timedelta(days=360))
with d_col2:
    a_start = st.date_input("After Start", value=today - timedelta(days=60))
    a_end = st.date_input("After End", value=today)
with d_col3:
    max_cloud = st.slider("Max Metadata Cloud (%)", min_value=0, max_value=80, value=20)
    scale_m = st.selectbox("Target Export Scale (m)", options=[10, 20, 30, 60], index=1)

b_col1, b_col2 = st.columns([2, 1])
with b_col1:
    selected_bands = st.multiselect(
        "Sentinel-2 Bands (COPERNICUS/S2_SR_HARMONIZED)",
        options=[b for b in SENTINEL2_BANDS.keys() if b != "SCL"],
        default=["B4", "B3", "B2", "B8"],
    )
with b_col2:
    apply_scl = st.checkbox("Apply Pixel-Level SCL Cloud/Shadow Mask", value=True)
    export_dest = st.selectbox(
        "Export Destination",
        options=["local_geotiff", "google_drive", "gcs"],
    )

st.caption(
    "Note: Metadata `CLOUDY_PIXEL_PERCENTAGE` is an image-level filter only; enabling pixel-level SCL masking "
    "excludes cloud/shadow pixels from composites and NDVI metrics."
)

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

act1, act2, act3 = st.columns(3)
disabled_gee = not gee_status.get("initialized")

with act1:
    if st.button("Search Available Scenes", use_container_width=True, disabled=disabled_gee):
        with st.spinner("Querying COPERNICUS/S2_SR_HARMONIZED catalog..."):
            try:
                st.session_state["gee_search_res"] = client.search_satellite(base_payload)
            except Exception as exc:
                st.error(f"Search error: {exc}")

with act2:
    if st.button("Preview Composites & NDVI", use_container_width=True, disabled=disabled_gee):
        with st.spinner("Building SCL-masked median composites and NDVI..."):
            try:
                st.session_state["gee_preview_res"] = client.preview_satellite(base_payload)
            except Exception as exc:
                st.error(f"Preview error: {exc}")

with act3:
    if st.button("Export Aligned Pair", type="primary", use_container_width=True, disabled=disabled_gee):
        with st.spinner("Exporting aligned Sentinel-2 GeoTIFF pair..."):
            try:
                exp_res = client.export_satellite(base_payload)
                st.session_state["gee_export_res"] = exp_res
                st.success(f"Export status: {exp_res.get('status')} (Task ID: `{exp_res.get('task_id')}`)")
            except Exception as exc:
                st.error(f"Export failed: {exc}")

search_res = st.session_state.get("gee_search_res")
if search_res:
    st.markdown("### 2. Candidate Sentinel-2 Observations")
    sc1, sc2 = st.columns(2)
    for col, key, label in ((sc1, "before", "Before Period"), (sc2, "after", "After Period")):
        with col:
            part = search_res.get(key, {})
            st.markdown(f"**{label}**")
            if part.get("empty"):
                st.warning(part.get("message"))
            elif part.get("scenes"):
                st.dataframe(pd.DataFrame(part["scenes"]), use_container_width=True, hide_index=True)

preview_res = st.session_state.get("gee_preview_res")
if preview_res:
    st.markdown("### 3. Masked Median Composites & NDVI Comparison")
    pc1, pc2 = st.columns(2)
    pc1.image(preview_res["before"]["thumb_url"], caption="Before Composite", use_container_width=True)
    pc2.image(preview_res["after"]["thumb_url"], caption="After Composite", use_container_width=True)
    with st.expander("Composite Resampling & Acquisition Metadata"):
        st.json({"before": preview_res["before"]["metadata"], "after": preview_res["after"]["metadata"]})

    ndvi = preview_res.get("ndvi")
    if ndvi and ndvi.get("available"):
        st.markdown("#### NDVI Vegetation Indicator")
        nc1, nc2, nc3 = st.columns(3)
        nc1.image(ndvi["before_ndvi_url"], caption="Before NDVI", use_container_width=True)
        nc2.image(ndvi["after_ndvi_url"], caption="After NDVI", use_container_width=True)
        nc3.image(ndvi["diff_ndvi_url"], caption="NDVI Difference (After - Before)", use_container_width=True)
        st.caption(ndvi.get("scientific_note", NDVI_SCIENTIFIC_NOTE))

export_res = st.session_state.get("gee_export_res")
if export_res:
    st.markdown("### 4. Export Status & Model Handoff")
    st.json(export_res)
    if export_res.get("destination") == "local_geotiff" and export_res.get("before_path"):
        if st.button("Send Prepared Pair to Change Detection", type="primary"):
            st.session_state["gee_prepared_pair"] = export_res
            st.success("Pair staged! Open **Change Detection** in the sidebar to run model inference.")
    elif export_res.get("task_id"):
        if st.button("Refresh Batch Task Status"):
            st.session_state["gee_export_res"] = client.get_satellite_task(export_res["task_id"])
            st.rerun()

