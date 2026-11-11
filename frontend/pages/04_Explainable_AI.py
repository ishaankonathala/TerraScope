"""Explainable AI page for ViT Attention Rollout, Swin Window Attention, and CNN Grad-CAM."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import setup_page
from ml.explainability import INTERPRETABILITY_DISCLAIMER

setup_page(
    "Explainable AI Visualizations",
    "Inspect ViT Attention Rollout, Swin Window Attention, and Siamese U-Net Grad-CAM overlays.",
    icon="🧠",
)

client = get_app_client()
st.info(INTERPRETABILITY_DISCLAIMER)

tab1, tab2 = st.tabs(["Inspect Saved Analysis", "Compare All 3 Models Live"])

with tab1:
    results = client.list_results(limit=25)
    if not results:
        st.warning("No analyzed pairs saved yet. Run an inference with XAI enabled on the Change Detection page.")
    else:
        options = [r["result_id"] for r in results]
        default_idx = 0
        if st.session_state.get("last_result_id") in options:
            default_idx = options.index(st.session_state["last_result_id"])
        rid = st.selectbox("Select Analysis Result", options=options, index=default_idx)
        res = client.get_result(rid)

        xai_meta = res.get("explainability") or {}
        st.write(f"**Model:** `{res.get('model_name')}` | **Method:** {xai_meta.get('method', 'Not generated')}")
        panel_bytes = client.get_artifact_bytes(rid, "xai_panel")
        overlay_bytes = client.get_artifact_bytes(rid, "xai_overlay")
        b_bytes = client.get_artifact_bytes(rid, "before_png")
        a_bytes = client.get_artifact_bytes(rid, "after_png")

        if panel_bytes:
            st.image(panel_bytes, caption="4-Panel Explainability Comparison", use_container_width=True)
        elif overlay_bytes:
            c1, c2, c3 = st.columns(3)
            with c1:
                if b_bytes:
                    st.image(b_bytes, caption="Before (T1)", use_container_width=True)
            with c2:
                if a_bytes:
                    st.image(a_bytes, caption="After (T2)", use_container_width=True)
            with c3:
                st.image(overlay_bytes, caption="XAI Overlay", use_container_width=True)
        else:
            st.warning("No XAI visualization was generated for this result. Enable XAI during prediction or use the live comparison tab.")

with tab2:
    st.markdown("Upload a before/after pair to inspect **Grad-CAM (Siamese U-Net)**, **Attention Rollout (ViT)**, and **Window Attention (Swin)** side by side.")
    c1, c2 = st.columns(2)
    with c1:
        up_b = st.file_uploader("Before Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="xai_b")
    with c2:
        up_a = st.file_uploader("After Image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="xai_a")

    if st.button("Generate Side-by-Side Explanations", type="primary"):
        if up_b is None or up_a is None:
            st.error("Upload both Before and After images.")
        else:
            cols = st.columns(3)
            for idx, (m_id, title) in enumerate(
                [
                    ("siamese_unet", "Siamese U-Net (Grad-CAM)"),
                    ("vit", "ViT (Attention Rollout)"),
                    ("swin", "Swin (Window Attention)"),
                ]
            ):
                with cols[idx]:
                    with st.spinner(f"Running {title}..."):
                        try:
                            out = client.predict_files(
                                before_name=up_b.name,
                                before_bytes=up_b.getvalue(),
                                after_name=up_a.name,
                                after_bytes=up_a.getvalue(),
                                model_name=m_id,
                                include_xai=True,
                            )
                            xai_img = client.get_artifact_bytes(out["result_id"], "xai_overlay")
                            mask_img = client.get_artifact_bytes(out["result_id"], "overlay")
                            st.markdown(f"**{title}**")
                            if xai_img:
                                st.image(xai_img, caption="Attention / Grad-CAM Overlay", use_container_width=True)
                            if mask_img:
                                st.image(
                                    mask_img,
                                    caption=f"Prediction Overlay ({out['change_percentage']:.2f}%)",
                                    use_container_width=True,
                                )
                        except Exception as exc:
                            st.error(f"{title} failed: {exc}")

