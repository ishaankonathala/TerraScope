"""TerraScope — Reports, Auditable Provenance & Multi-Format Exports.

Generates research-grade PDF summaries, structured CSV metrics, JSON provenance records,
and GIS vector/raster outputs for bi-temporal satellite change analyses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

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

setup_page(
    "Reports & Research Exports",
    "Export publication-ready PDF reports, structured CSV metrics, full JSON provenance logs, and GIS vector/raster assets.",
    icon="📄",
)

colors = get_theme_colors()
client = get_app_client()
results = client.list_results(limit=50)

if not results:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px dashed {colors['border']}; border-radius: 8px; padding: 32px; text-align: center;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 6px;">
                No Analysis Reports Available
            </div>
            <p style="color: {colors['muted']}; font-size: 0.92rem; margin: 0;">
                Perform an inference run in the <strong>Change Detection</strong> studio to generate auditable research reports.
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
    rid = st.selectbox("Select Target Analysis Record", options=options, index=default_idx)
    res = client.get_result(rid)
    geo = res.get("geospatial", {})

with col_stat:
    is_geo = bool(geo.get("is_georeferenced"))
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 14px 18px; margin-top: 24px;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['sage']}; font-weight: 700; margin-bottom: 4px;">
                Audit Fingerprint
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {colors['forest']}; word-break: break-all;">
                {rid}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# High-level metrics
render_metric_cards(
    [
        {"title": "Neural Model", "value": res.get("model_name", "Unknown")},
        {"title": "Detection Ratio", "value": f"{res.get('change_percentage', 0.0):.2f}%"},
        {"title": "Changed Pixels", "value": f"{res.get('changed_pixels', 0):,}"},
        {
            "title": "Ground Area",
            "value": f"{geo.get('changed_area_ha', 0.0):.4f} ha" if is_geo else "Image-Relative",
            "sub": str(geo.get("crs", "Unprojected")) if is_geo else "No CRS",
        },
    ]
)

st.write("")

# Detailed Summary Table
st.markdown(
    f"""
    <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
        Deterministic Audit Provenance
    </h4>
    """,
    unsafe_allow_html=True,
)

audit_rows = [
    {"Field": "Analysis Identifier", "Value": res["result_id"]},
    {"Field": "Architecture Backbone", "Value": res.get("model_name")},
    {"Field": "Weights & Training Status", "Value": res.get("training_status", "Baseline Mode")},
    {"Field": "Decision Threshold", "Value": f"{res.get('threshold', 0.5):.2f}"},
    {"Field": "Execution Latency", "Value": f"{res.get('inference_time_sec', 0.0):.3f} seconds"},
    {"Field": "Changed Pixels", "Value": f"{res.get('changed_pixels', 0):,}"},
    {"Field": "Valid Monitored Pixels", "Value": f"{res.get('total_valid_pixels', 0):,}"},
    {"Field": "Invalid / Masked Pixels", "Value": f"{res.get('masked_invalid_pixels', 0):,}"},
    {"Field": "Change Proportion", "Value": f"{res.get('change_percentage', 0.0):.4f}%"},
    {"Field": "Coordinate Reference System", "Value": geo.get("crs") or "None (Image-relative grid)"},
    {
        "Field": "Physical Changed Area",
        "Value": (
            f"{geo['changed_area_ha']:.4f} ha ({geo['changed_area_m2']:,.1f} m²)"
            if geo.get("changed_area_ha") is not None
            else "Not applicable (unprojected)"
        ),
    },
]
st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)

st.write("")
render_research_note(
    "Auditable Research Standard: Every exported artifact (PDF, CSV, GeoTIFF, GeoJSON) contains matching deterministic timestamps and cryptographic parameters to ensure reproducibility."
)
st.write("")

# Download Grid
st.markdown(
    f"""
    <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 12px;">
        Download Research Artifacts
    </h4>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    pdf_b = client.get_artifact_bytes(rid, "pdf")
    if pdf_b:
        st.download_button(
            "Download PDF Report",
            data=pdf_b,
            file_name=f"{rid}_terrascope_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    else:
        st.button("PDF Generating...", disabled=True, use_container_width=True)

with c2:
    csv_b = client.get_artifact_bytes(rid, "csv")
    if csv_b:
        st.download_button(
            "Download CSV Metrics",
            data=csv_b,
            file_name=f"{rid}_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )

with c3:
    mask_b = client.get_artifact_bytes(rid, "png")
    if mask_b:
        st.download_button(
            "Download Binary Mask (PNG)",
            data=mask_b,
            file_name=f"{rid}_binary_mask.png",
            mime="image/png",
            use_container_width=True,
        )

with c4:
    ov_b = client.get_artifact_bytes(rid, "overlay")
    if ov_b:
        st.download_button(
            "Download Overlay (PNG)",
            data=ov_b,
            file_name=f"{rid}_change_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

st.write("")

c5, c6, c7, c8 = st.columns(4)

with c5:
    json_b = client.get_artifact_bytes(rid, "json")
    if json_b:
        st.download_button(
            "Download Provenance (JSON)",
            data=json_b,
            file_name=f"{rid}_provenance.json",
            mime="application/json",
            use_container_width=True,
        )

with c6:
    xai_b = client.get_artifact_bytes(rid, "xai_panel")
    if xai_b:
        st.download_button(
            "Download XAI Panel (PNG)",
            data=xai_b,
            file_name=f"{rid}_explainability.png",
            mime="image/png",
            use_container_width=True,
        )
    else:
        st.button("XAI Panel (N/A)", disabled=True, use_container_width=True)

with c7:
    tif_b = client.get_artifact_bytes(rid, "geotiff")
    if tif_b:
        st.download_button(
            "Download Mask (GeoTIFF)",
            data=tif_b,
            file_name=f"{rid}_change_mask.tif",
            mime="image/tiff",
            use_container_width=True,
        )
    else:
        st.button("GeoTIFF Mask (N/A)", disabled=True, use_container_width=True)

with c8:
    gj_b = client.get_artifact_bytes(rid, "geojson")
    if gj_b:
        st.download_button(
            "Download Polygons (GeoJSON)",
            data=gj_b,
            file_name=f"{rid}_polygons.geojson",
            mime="application/geo+json",
            use_container_width=True,
        )
    else:
        st.button("GeoJSON Polygons (N/A)", disabled=True, use_container_width=True)
