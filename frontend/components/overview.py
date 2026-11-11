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
    experiments = client.get_experiments()
    results = client.list_results(limit=50)

    # Sidebar System Telemetry
    api_online = client.is_api_online()
    with st.sidebar:
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; margin-bottom: 8px;">
                <div style="color: {colors['text_muted']};">Backend Engine</div>
                <div style="font-weight: 600; color: {colors['text']};">
                    {'● FastAPI Live (Port 8000/8001)' if api_online else '○ Direct Local Python Runtime'}
                </div>
            </div>
            <div style="font-size: 0.8rem; margin-bottom: 8px;">
                <div style="color: {colors['text_muted']};">Compute Hardware</div>
                <div style="font-weight: 600; color: {colors['text']};">
                    {health['device']['selected_device'].upper()} (PyTorch {health['device']['torch_version']})
                </div>
            </div>
            <div style="font-size: 0.8rem; margin-bottom: 12px;">
                <div style="color: {colors['text_muted']};">Earth Engine Status</div>
                <div style="font-weight: 600; color: {colors['text']};">
                    {health['gee_status']['status_code'].upper()}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

    # Executive Hero Banner / Call to Action
    st.markdown(
        f"""
        <div style="background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 22px 26px; margin-bottom: 22px; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px;">
            <div style="max-width: 680px;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: {colors['text']}; margin-bottom: 6px;">
                    Monitor Environmental and Urban Transitions
                </div>
                <div style="font-size: 0.9rem; color: {colors['text_muted']}; line-height: 1.5;">
                    Compare satellite scenes captured at two different timestamps. TerraScope pairs high-resolution imagery, calculates pixel-level change probabilities, isolates valid cloud-free surfaces, and derives real-world spatial statistics.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick Start Navigation Cards
    col_act1, col_act2, col_act3 = st.columns(3)
    with col_act1:
        if st.button("🚀 Launch Change Detection", type="primary", use_container_width=True):
            st.switch_page("pages/02_Change_Detection.py")
    with col_act2:
        if st.button("🌍 Explore Satellite Imagery (GEE)", use_container_width=True):
            st.switch_page("pages/07_Satellite_Data_Explorer.py")
    with col_act3:
        if st.button("📊 Review Model Benchmarks", use_container_width=True):
            st.switch_page("pages/03_Model_Comparison.py")

    # High-level Metrics Strip
    total_pairs = len(results)
    total_changed_px = sum(int(r.get("changed_pixels", 0)) for r in results)
    geo_areas = [
        float(r["geospatial"]["changed_area_ha"])
        for r in results
        if r.get("geospatial", {}).get("changed_area_ha") is not None
    ]
    trained_count = sum(1 for m in models_data["models"] if m["has_trained_checkpoint"])

    render_metric_cards(
        [
            {
                "title": "Analyzed Pairs",
                "value": f"{total_pairs}",
                "sub": "Logged in outputs/change_maps",
            },
            {
                "title": "Detected Change",
                "value": f"{sum(geo_areas):.2f} ha" if geo_areas else f"{total_changed_px:,} px",
                "sub": "Hectares (georeferenced) or valid px",
            },
            {
                "title": "Supported Architectures",
                "value": "3 Models",
                "sub": f"{trained_count} fine-tuned checkpoints available",
            },
            {
                "title": "Hardware Accelerator",
                "value": health["device"]["selected_device"].upper(),
                "sub": "Apple Silicon / CPU auto-detected",
            },
        ]
    )

    # Main Grid: Model Zoo & Recent Activity
    col_left, col_right = st.columns([1.15, 1.0])

    with col_left:
        st.markdown("### Neural Architecture Zoo")
        st.caption("Three fundamentally distinct deep learning paradigms for multi-temporal remote sensing.")

        models_list = [
            {
                "Model": "Siamese U-Net",
                "Type": "Convolutional (CNN)",
                "Focus": "Local texture, crisp edges, fast inference",
                "Status": "Checkpoint Ready" if any(c["filename"].startswith("siamese_unet") for c in models_data.get("checkpoints", [])) else "Untrained Head",
                "XAI Method": "Grad-CAM",
            },
            {
                "Model": "Siamese ViT",
                "Type": "Vision Transformer",
                "Focus": "Global spatial context, long-range tokens",
                "Status": "Checkpoint Ready" if any(c["filename"].startswith("vit") for c in models_data.get("checkpoints", [])) else "Untrained Head",
                "XAI Method": "Attention Rollout",
            },
            {
                "Model": "Siamese Swin",
                "Type": "Hierarchical Transformer",
                "Focus": "Multi-scale shifted windows, dense boundaries",
                "Status": "Checkpoint Ready" if any(c["filename"].startswith("swin") for c in models_data.get("checkpoints", [])) else "Untrained Head",
                "XAI Method": "Window Attention",
            },
        ]
        st.dataframe(pd.DataFrame(models_list), use_container_width=True, hide_index=True)

        render_research_note(
            "Scientific Integrity Notice",
            "Models without trained checkpoints run with initialized backbone representations for pipeline validation. Metrics are strictly recorded as 'Not evaluated' until full training on official benchmarks (LEVIR-CD / OSCD) is executed.",
        )

    with col_right:
        st.markdown("### Recent Change Analyses")
        st.caption("Persisted observation records stored in the workspace.")

        if not results:
            st.info("No analyses recorded yet. Run a prediction on the **Change Detection** page.")
        else:
            rec_rows = []
            for r in results[:8]:
                geo = r.get("geospatial", {})
                rec_rows.append(
                    {
                        "Result ID": r["result_id"],
                        "Model": r["model_name"],
                        "Change": f"{r['change_percentage']:.2f}%",
                        "Area (ha)": f"{geo['changed_area_ha']:.2f}" if geo.get("changed_area_ha") is not None else "Image-relative",
                        "Time": f"{r.get('inference_time_sec', 0):.2f}s",
                    }
                )
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

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
