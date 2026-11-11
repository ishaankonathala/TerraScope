"""Reports & Exports page for CSV metrics, GeoTIFFs, visualizations, and PDF reports."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import setup_page

setup_page(
    "Reports & Research Exports",
    "Download structured CSV metrics, JSON provenance logs, visualizations, GeoTIFFs, and PDF reports.",
    icon="📄",
)

client = get_app_client()
results = client.list_results(limit=50)

if not results:
    st.info("No prediction reports exist yet. Analyze an image pair on **Change Detection** first.")
    st.stop()

options = [r["result_id"] for r in results]
default_idx = 0
if st.session_state.get("last_result_id") in options:
    default_idx = options.index(st.session_state["last_result_id"])

rid = st.selectbox("Select Analysis Result", options=options, index=default_idx)
res = client.get_result(rid)
geo = res.get("geospatial", {})

summary_df = pd.DataFrame(
    [
        {"Metric": "Result ID", "Value": res["result_id"]},
        {"Metric": "Model", "Value": res["model_name"]},
        {"Metric": "Training Status", "Value": res.get("training_status")},
        {"Metric": "Threshold", "Value": res["threshold"]},
        {"Metric": "Changed Pixels", "Value": f"{res['changed_pixels']:,}"},
        {"Metric": "Total Valid Pixels", "Value": f"{res['total_valid_pixels']:,}"},
        {"Metric": "Masked Pixels", "Value": f"{res['masked_invalid_pixels']:,}"},
        {"Metric": "Change Percentage", "Value": f"{res['change_percentage']:.4f}%"},
        {"Metric": "CRS", "Value": geo.get("crs") or "Not georeferenced"},
        {
            "Metric": "Changed Area (ha)",
            "Value": (
                f"{geo['changed_area_ha']:.4f}"
                if geo.get("changed_area_ha") is not None
                else "N/A (image-relative)"
            ),
        },
    ]
)
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.markdown("### Download Artifacts")
cols = st.columns(4)
downloads = [
    ("CSV Metrics", "csv", f"{rid}_metrics.csv", "text/csv"),
    ("PDF Report", "pdf", f"{rid}_report.pdf", "application/pdf"),
    ("Change Mask PNG", "png", f"{rid}_mask.png", "image/png"),
    ("Overlay PNG", "overlay", f"{rid}_overlay.png", "image/png"),
]
for col, (label, fmt, fname, mime) in zip(cols, downloads):
    with col:
        data = client.get_artifact_bytes(rid, fmt)
        if data:
            st.download_button(label, data=data, file_name=fname, mime=mime, use_container_width=True)

cols2 = st.columns(4)
extra_downloads = [
    ("Full JSON", "json", f"{rid}.json", "application/json"),
    ("XAI Panel PNG", "xai_panel", f"{rid}_xai.png", "image/png"),
    ("GeoTIFF Mask", "geotiff", f"{rid}_change_mask.tif", "image/tiff"),
    ("GeoJSON Polygons", "geojson", f"{rid}_regions.geojson", "application/geo+json"),
]
for col, (label, fmt, fname, mime) in zip(cols2, extra_downloads):
    with col:
        data = client.get_artifact_bytes(rid, fmt)
        if data:
            st.download_button(label, data=data, file_name=fname, mime=mime, use_container_width=True)

