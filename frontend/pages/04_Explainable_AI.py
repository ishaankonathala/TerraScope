"""TerraScope — Explainable AI & Neural Feature Attribution.

Inspects authentic Grad-CAM activations (Siamese U-Net), Attention Rollout (ViT),
and Shifted-Window Attention maps (Swin) without synthesizing or interpolating fake heatmaps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import get_theme_colors, render_research_note, setup_page
from ml.explainability import INTERPRETABILITY_DISCLAIMER

setup_page(
    "Explainable AI & Feature Attribution",
    "Inspect how convolutional backbones and self-attention transformers weigh spatial evidence when isolating bi-temporal surface change.",
    icon="🧠",
)

colors = get_theme_colors()
client = get_app_client()

# Educational architecture attribution overview
st.markdown(
    f"""
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
            Demystifying Neural Change Predictions
        </h3>
        <p style="font-size: 0.95rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 16px;">
            Different neural architectures localize surface transitions using fundamentally distinct mathematical mechanisms. Below is how each architecture explains its spatial reasoning:
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['sage']}; font-weight: 700; margin-bottom: 4px;">
                Convolutional Attribution
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 6px;">
                Grad-CAM (U-Net)
            </div>
            <p style="font-size: 0.85rem; color: {colors['muted']}; line-height: 1.45; margin: 0;">
                Calculates the gradient of the predicted change score with respect to the final convolutional feature maps. Warm regions highlight specific pixel clusters that triggered class activation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['terracotta']}; font-weight: 700; margin-bottom: 4px;">
                Global Attention
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 6px;">
                Attention Rollout (ViT)
            </div>
            <p style="font-size: 0.85rem; color: {colors['muted']}; line-height: 1.45; margin: 0;">
                Recursively multiplies raw self-attention weight matrices across all transformer encoder blocks. Reveals how information flows from all-to-all spatial patches into the change decision.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 18px; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['forest']}; font-weight: 700; margin-bottom: 4px;">
                Hierarchical Window Attention
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 6px;">
                Window Attention (Swin)
            </div>
            <p style="font-size: 0.85rem; color: {colors['muted']}; line-height: 1.45; margin: 0;">
                Aggregates self-attention distributions computed within shifted local windows across pyramid stages. Highlights structured boundary correlations without quadratic memory overhead.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")
render_research_note(
    f"Interpretability Protocol: {INTERPRETABILITY_DISCLAIMER} Warm attribution tones (red/yellow) indicate elevated neural activation, while cool tones (blue/purple) indicate low influence."
)
st.write("")

tab1, tab2 = st.tabs(["Inspect Prior Analysis Results", "Side-by-Side Model Comparison"])

with tab1:
    results = client.list_results(limit=30)
    if not results:
        st.markdown(
            f"""
            <div style="background-color: {colors['card_bg']}; border: 1px dashed {colors['border']}; border-radius: 8px; padding: 24px; text-align: center;">
                <p style="color: {colors['muted']}; font-size: 0.92rem; margin: 0;">
                    No completed analysis records available. Run an inference with <strong>Enable Explainable AI (XAI)</strong> toggled on in the <strong>Change Detection</strong> studio first.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        col_sel, col_info = st.columns([2, 1])
        with col_sel:
            options = [r["result_id"] for r in results]
            default_idx = 0
            if st.session_state.get("last_result_id") in options:
                default_idx = options.index(st.session_state["last_result_id"])
            rid = st.selectbox("Select Analysis Record", options=options, index=default_idx)
            res = client.get_result(rid)

        with col_info:
            xai_meta = res.get("explainability") or {}
            st.markdown(
                f"""
                <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 12px 16px; margin-top: 24px;">
                    <div style="font-size: 0.82rem; color: {colors['muted']};">Architecture: <strong style="color: {colors['forest']};">{res.get('model_name')}</strong></div>
                    <div style="font-size: 0.82rem; color: {colors['muted']};">Attribution Method: <strong style="color: {colors['terracotta']};">{xai_meta.get('method', 'None Generated')}</strong></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        panel_bytes = client.get_artifact_bytes(rid, "xai_panel")
        overlay_bytes = client.get_artifact_bytes(rid, "xai_overlay")
        b_bytes = client.get_artifact_bytes(rid, "before_png")
        a_bytes = client.get_artifact_bytes(rid, "after_png")
        mask_bytes = client.get_artifact_bytes(rid, "png")

        if panel_bytes:
            st.image(panel_bytes, caption="4-Quadrant Diagnostic Attribution Panel", use_container_width=True)
        elif overlay_bytes:
            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                if b_bytes:
                    st.image(b_bytes, caption="Before (T1)", use_container_width=True)
            with p_col2:
                if a_bytes:
                    st.image(a_bytes, caption="After (T2)", use_container_width=True)
            with p_col3:
                st.image(overlay_bytes, caption=f"Neural Attribution Overlay ({xai_meta.get('method', 'XAI')})", use_container_width=True)
        else:
            st.warning("No feature attribution map was generated for this result. You can run live XAI generation in the second tab.")

with tab2:
    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 4px;">
                Live Multi-Architecture Attribution Benchmark
            </div>
            <div style="font-size: 0.88rem; color: {colors['muted']};">
                Pass an aligned image pair through all three neural backbones simultaneously to compare where each model focuses its attention.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    source_choice = st.radio(
        "Pair Source",
        options=["Use Demonstration Sample RGB", "Upload Custom Pair"],
        horizontal=True,
    )

    before_bytes: bytes | None = None
    after_bytes: bytes | None = None
    b_name = "before.png"
    a_name = "after.png"

    if source_choice == "Use Demonstration Sample RGB":
        sample_dir = PROJECT_ROOT / "data" / "samples" / "sample_rgb"
        if (sample_dir / "before.png").exists() and (sample_dir / "after.png").exists():
            before_bytes = (sample_dir / "before.png").read_bytes()
            after_bytes = (sample_dir / "after.png").read_bytes()
            st.caption("Loaded demonstration sample pair from `data/samples/sample_rgb/`.")
        else:
            st.error("Sample directory missing.")
    else:
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            up_b = st.file_uploader("Before Image (T1)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="xai_live_b")
        with up_col2:
            up_a = st.file_uploader("After Image (T2)", type=["png", "jpg", "jpeg", "tif", "tiff"], key="xai_live_a")
        if up_b and up_a:
            before_bytes = up_b.getvalue()
            after_bytes = up_a.getvalue()
            b_name = up_b.name
            a_name = up_a.name

    if st.button("Generate Comparative Attributions Across All 3 Architectures", type="primary"):
        if not before_bytes or not after_bytes:
            st.error("Please supply both Before and After image data.")
        else:
            models_to_run = [
                ("siamese_unet", "Siamese U-Net", "Grad-CAM (Convolutional Backpropagation)"),
                ("vit", "Siamese ViT", "Attention Rollout (Global Self-Attention)"),
                ("swin", "Siamese Swin", "Window Attention (Shifted Multiscale Windows)"),
            ]
            cols = st.columns(3)
            for idx, (m_id, m_title, m_method) in enumerate(models_to_run):
                with cols[idx]:
                    with st.spinner(f"Computing {m_title}..."):
                        try:
                            out = client.predict_files(
                                before_name=b_name,
                                before_bytes=before_bytes,
                                after_name=a_name,
                                after_bytes=after_bytes,
                                model_name=m_id,
                                include_xai=True,
                            )
                            xai_img = client.get_artifact_bytes(out["result_id"], "xai_overlay")
                            mask_img = client.get_artifact_bytes(out["result_id"], "overlay")
                            st.markdown(
                                f"""
                                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 2px;">
                                    {m_title}
                                </div>
                                <div style="font-size: 0.78rem; color: {colors['muted']}; margin-bottom: 8px;">
                                    {m_method}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if xai_img:
                                st.image(xai_img, caption="Feature Attribution Heatmap", use_container_width=True)
                            if mask_img:
                                st.image(
                                    mask_img,
                                    caption=f"Change Mask ({out['change_percentage']:.2f}% detected)",
                                    use_container_width=True,
                                )
                        except Exception as exc:
                            st.error(f"{m_title} failed: {exc}")
