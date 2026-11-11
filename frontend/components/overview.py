"""TerraScope Overview Dashboard — executive platform summary and quick actions."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from frontend.components.client import get_app_client
from frontend.components.theme import (
    get_theme_colors,
    render_metric_cards,
    render_research_note,
    setup_page,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def render_overview_dashboard() -> None:
    theme = setup_page(
        "Platform Overview",
        "TerraScope detects, measures, and visualizes surface changes across bi-temporal Earth observation imagery using deep convolutional and vision transformer backbones.",
        icon="🌍",
    )
    colors = get_theme_colors(theme)

    client = get_app_client()
    health = client.get_health()
    models_data = client.get_models()
    results = client.list_results(limit=50)

    # Sidebar System Telemetry
    api_online = client.is_api_online()
    with st.sidebar:
        st.caption("SYSTEM RUNTIME")
        engine_label = "● FastAPI Live (Port 8000/8001)" if api_online else "○ Direct Local Runtime"
        device_label = f"{health['device']['selected_device'].upper()} (PyTorch {health['device']['torch_version']})"
        gee_label = health["gee_status"]["status_code"].upper()

        telemetry_html = (
            f'<div style="font-size: 0.82rem; margin-bottom: 6px;">'
            f'<span style="color: {colors["text_muted"]};">Engine:</span> '
            f'<strong style="color: {colors["text"]};">{engine_label}</strong>'
            f'</div>'
            f'<div style="font-size: 0.82rem; margin-bottom: 6px;">'
            f'<span style="color: {colors["text_muted"]};">Compute:</span> '
            f'<strong style="color: {colors["text"]};">{device_label}</strong>'
            f'</div>'
            f'<div style="font-size: 0.82rem; margin-bottom: 12px;">'
            f'<span style="color: {colors["text_muted"]};">Earth Engine:</span> '
            f'<strong style="color: {colors["text"]};">{gee_label}</strong>'
            f'</div>'
        )
        st.markdown(telemetry_html, unsafe_allow_html=True)
        st.divider()

    # Quick Start Navigation Buttons
    col_act1, col_act2, col_act3 = st.columns(3, gap="medium")
    with col_act1:
        if st.button("🚀  Launch Change Detection", type="primary", use_container_width=True):
            st.switch_page("pages/02_Change_Detection.py")
    with col_act2:
        if st.button("🌍  Explore Satellite Data (GEE)", use_container_width=True):
            st.switch_page("pages/07_Satellite_Data_Explorer.py")
    with col_act3:
        if st.button("📊  Model Benchmark Registry", use_container_width=True):
            st.switch_page("pages/03_Model_Comparison.py")

    st.write("")

    # High-level Metrics Strip
    total_pairs = len(results)
    total_changed_px = sum(int(r.get("changed_pixels", 0)) for r in results)
    geo_areas = [
        float(r["geospatial"]["changed_area_ha"])
        for r in results
        if r.get("geospatial", {}).get("changed_area_ha") is not None
    ]
    trained_count = sum(1 for m in models_data.get("models", []) if m.get("has_trained_checkpoint"))

    render_metric_cards(
        [
            {
                "title": "Analyzed Pairs",
                "value": f"{total_pairs}",
                "sub": "Stored in outputs",
            },
            {
                "title": "Detected Change",
                "value": f"{sum(geo_areas):.2f} ha" if geo_areas else f"{total_changed_px:,} px",
                "sub": "Georeferenced area" if geo_areas else "Valid pixels",
            },
            {
                "title": "Neural Models",
                "value": "3 Architectures",
                "sub": f"{trained_count} trained weights",
            },
            {
                "title": "Compute Device",
                "value": health["device"]["selected_device"].upper(),
                "sub": f"PyTorch {health['device']['torch_version']}",
            },
        ]
    )

    st.write("")

    # Main Grid: Model Zoo & Recent Activity
    col_left, col_right = st.columns([1.1, 1.0], gap="large")

    with col_left:
        st.markdown("#### Neural Architecture Zoo")
        st.caption("Empirical comparison of convolutional and self-attention paradigms for remote sensing.")

        models_list = [
            {
                "Architecture": "Siamese U-Net",
                "Family": "Convolutional (CNN)",
                "Spatial Focus": "Local texture, edge delineation",
                "Attribution": "Grad-CAM",
            },
            {
                "Architecture": "Siamese ViT",
                "Family": "Vision Transformer",
                "Spatial Focus": "Global context, long-range tokens",
                "Attribution": "Attention Rollout",
            },
            {
                "Architecture": "Siamese Swin",
                "Family": "Hierarchical Transformer",
                "Spatial Focus": "Shifted windows, dense boundaries",
                "Attribution": "Window Attention",
            },
        ]
        st.dataframe(pd.DataFrame(models_list), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("#### Recent Change Analyses")
        st.caption("Latest bi-temporal observation records stored in workspace.")

        if not results:
            st.info("No analyses recorded yet. Run a prediction on the **Change Detection** page.")
        else:
            rec_rows = []
            for r in results[:6]:
                geo = r.get("geospatial", {})
                area_val = f"{geo['changed_area_ha']:.2f} ha" if geo.get("changed_area_ha") is not None else "Image-relative"
                rec_rows.append(
                    {
                        "Result ID": r["result_id"],
                        "Model": r.get("model_name", "Unknown"),
                        "Change": f"{r.get('change_percentage', 0.0):.2f}%",
                        "Area": area_val,
                        "Latency": f"{r.get('inference_time_sec', 0):.2f}s",
                    }
                )
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

    st.write("")
    render_research_note(
        "Scientific Integrity Notice",
        "Models without fine-tuned checkpoints run with initialized backbone representations for pipeline validation. Metrics are strictly recorded as 'Not evaluated' until full training on official benchmarks (LEVIR-CD / OSCD) is executed.",
    )

    # Core Research Questions
    with st.expander("TerraScope Research Questions (RQ1 – RQ4)", expanded=False):
        st.markdown(
            """
            - **RQ1 (Architectural Efficacy):** Do self-attention Vision Transformers yield statistically significant improvements over convolutional Siamese baselines on high-resolution satellite imagery?
            - **RQ2 (Boundary vs Context):** Does Swin Transformer's shifted window hierarchy provide sharper structural boundaries than ViT's uniform patch tokenization?
            - **RQ3 (Environmental Resilience):** How robust are the models against seasonal phenology deltas, atmospheric haze, and SCL cloud masking?
            - **RQ4 (Feature Attribution):** Does Attention Rollout and Swin Window Attention correlate more closely with human ground-truth annotations than CNN Grad-CAM?
            """
        )
