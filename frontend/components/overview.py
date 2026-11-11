"""Shared Overview Dashboard rendering logic."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components.client import get_app_client
from frontend.components.theme import render_metric_cards, setup_page


def render_overview_dashboard() -> None:
    setup_page(
        "Environmental Monitoring Overview",
        "Multi-temporal satellite change detection comparing CNN, Vision Transformer, and Swin Transformer models.",
        icon="🛰️",
    )

    client = get_app_client()
    health = client.get_health()
    models_data = client.get_models()
    experiments = client.get_experiments()
    results = client.list_results(limit=50)

    api_mode = "FastAPI Live" if client.is_api_online() else "Direct Local Runtime"
    with st.sidebar:
        st.caption(f"Backend Mode: **{api_mode}**")
        st.caption(f"Compute Device: **{health['device']['selected_device'].upper()}**")
        gee_code = health["gee_status"]["status_code"]
        st.caption(f"Google Earth Engine: **{gee_code}**")

    total_pairs = len(results)
    total_changed_px = sum(int(r.get("changed_pixels", 0)) for r in results)
    geo_areas = [
        float(r["geospatial"]["changed_area_ha"])
        for r in results
        if r.get("geospatial", {}).get("changed_area_ha") is not None
    ]
    trained_models = sum(1 for m in models_data["models"] if m["has_trained_checkpoint"])
    selected_model = st.session_state.get("selected_model", "siamese_unet")

    render_metric_cards(
        [
            {
                "title": "Analyzed Image Pairs",
                "value": f"{total_pairs}",
                "sub": "Recorded in outputs/change_maps",
            },
            {
                "title": "Total Changed Area / Pixels",
                "value": f"{sum(geo_areas):.2f} ha" if geo_areas else f"{total_changed_px:,} px",
                "sub": "Real-world ha (georeferenced) or valid px",
            },
            {
                "title": "Active Model",
                "value": selected_model,
                "sub": f"{trained_models}/3 architectures with saved checkpoints",
            },
            {
                "title": "Compute Runtime",
                "value": health["device"]["selected_device"].upper(),
                "sub": f"PyTorch {health['device']['torch_version']}",
            },
        ]
    )

    st.markdown("### Architecture Availability & Provenance")
    arch_rows = []
    for m in models_data["models"]:
        arch_rows.append(
            {
                "Model ID": m["model_id"],
                "Architecture": m["display_name"],
                "Default Backbone": m["default_backbone"],
                "Trained Checkpoints": len(m["available_checkpoints"]),
                "Status": "Checkpoint Ready" if m["has_trained_checkpoint"] else "Awaiting Training",
                "Explainability": "Grad-CAM" if m["supports_gradcam"] else "Attention / Window Maps",
            }
        )
    st.dataframe(pd.DataFrame(arch_rows), use_container_width=True, hide_index=True)

    col_left, col_right = st.columns([1.2, 1.0])
    with col_left:
        st.markdown("### Recent Change Analyses")
        if not results:
            st.info(
                "No image pairs have been analyzed yet. Open **Change Detection** or "
                "**Satellite Data Explorer** in the sidebar to run your first prediction."
            )
        else:
            rec_rows = []
            for r in results[:10]:
                geo = r.get("geospatial", {})
                rec_rows.append(
                    {
                        "Result ID": r["result_id"],
                        "Model": r["model_name"],
                        "Status": r.get("training_status", "unknown"),
                        "Change (%)": f"{r['change_percentage']:.2f}%",
                        "Changed Area": (
                            f"{geo['changed_area_ha']:.4f} ha"
                            if geo.get("changed_area_ha") is not None
                            else "Image-relative"
                        ),
                        "Time (s)": r["inference_time_sec"],
                    }
                )
            st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

    with col_right:
        st.markdown("### Logged Benchmark Summary")
        comp_df = pd.DataFrame(experiments["comparison_table"])
        st.dataframe(
            comp_df[["display_name", "status", "dataset", "f1_score", "iou"]],
            use_container_width=True,
            hide_index=True,
        )
        if experiments["total_real_experiments"] == 0:
            st.caption(
                "No benchmark evaluations have been logged yet (displaying 'Not evaluated' rather than fabricated numbers)."
            )

    with st.expander("Research Questions Investigated (RQ1–RQ4)", expanded=False):
        st.markdown(
            """
            - **RQ1:** How do Vision Transformer-based architectures (`ViT`, `Swin Transformer`) compare with traditional CNN architectures (`Siamese U-Net`) under identical splits and preprocessing?
            - **RQ2:** How effectively do global self-attention, shifted-window hierarchical attention, and convolutional skip connections capture spatial changes?
            - **RQ3:** How do seasonal differences, cloud interference, and image misalignment impact change-detection accuracy?
            - **RQ4:** Can Attention Rollout, Swin Window Attention, and Grad-CAM provide faithful interpretability for predicted land-use changes?
            """
        )

