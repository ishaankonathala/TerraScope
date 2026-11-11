"""Change Detection interactive upload, inference, visualization, and download page."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import render_metric_cards, setup_page

setup_page(
    "Bi-Temporal Change Detection",
    "Upload or import aligned satellite pairs, run Siamese CNN/ViT/Swin inference, and export results.",
    icon="🔍",
)

client = get_app_client()
models_data = client.get_models()
model_options = [m["model_id"] for m in models_data["models"]]
model_labels = {m["model_id"]: m["display_name"] for m in models_data["models"]}
ckpt_by_model = {
    m["model_id"]: [c["filename"] for c in m["available_checkpoints"]]
    for m in models_data["models"]
}

gee_pair = st.session_state.get("gee_prepared_pair")
if gee_pair:
    st.info(
        f"Prepared Google Earth Engine Sentinel-2 pair loaded: **{gee_pair.get('task_id')}** "
        f"({gee_pair.get('before_path')}, {gee_pair.get('after_path')})."
    )
    if st.button("Clear GEE Prepared Pair"):
        st.session_state.pop("gee_prepared_pair", None)
        st.rerun()

col_up1, col_up2 = st.columns(2)
with col_up1:
    before_upload = st.file_uploader(
        "Before Satellite Image (T1)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="cd_before_upload",
    )
with col_up2:
    after_upload = st.file_uploader(
        "After Satellite Image (T2)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="cd_after_upload",
    )

with st.expander("Model & Inference Parameters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_model = st.selectbox(
            "Architecture",
            options=model_options,
            format_func=lambda k: model_labels.get(k, k),
        )
        st.session_state["selected_model"] = selected_model
        available_ckpts = ckpt_by_model.get(selected_model, [])
        ckpt_choice = st.selectbox(
            "Checkpoint",
            options=["None (backbone / untrained head)"] + available_ckpts,
        )
        checkpoint_name = None if ckpt_choice.startswith("None") else ckpt_choice
    with c2:
        threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
        norm_mode = st.selectbox(
            "Normalization",
            options=["imagenet", "sentinel2_sr", "unit", "zscore", "percentile"],
            index=1 if gee_pair else 0,
        )
    with c3:
        tile_size = st.selectbox("Tile Size (px)", options=[128, 256, 512], index=1)
        stride = st.selectbox("Tile Stride (px)", options=[96, 128, 192, 256], index=2)

    p1, p2, p3 = st.columns(3)
    with p1:
        post_enabled = st.checkbox("Enable Post-Processing", value=False)
    with p2:
        morph_op = st.selectbox(
            "Morphological Filter",
            options=["none", "open", "close", "open_close"],
            disabled=not post_enabled,
        )
    with p3:
        min_region = st.number_input(
            "Min Region Size (px)",
            min_value=0,
            max_value=5000,
            value=16 if post_enabled else 0,
            disabled=not post_enabled,
        )
    include_xai = st.checkbox("Generate Explainability Heatmap (Attention / Grad-CAM)", value=True)

if checkpoint_name is None:
    st.warning(
        f"No trained change-detection checkpoint selected for `{selected_model}`. "
        "Predictions reflect backbone initialization/untrained decoder head for pipeline testing only."
    )

run_clicked = st.button("Run Change Detection", type="primary", use_container_width=True)

if run_clicked:
    before_bytes: bytes | None = None
    after_bytes: bytes | None = None
    before_name = "before.png"
    after_name = "after.png"

    if before_upload is not None and after_upload is not None:
        before_bytes = before_upload.getvalue()
        after_bytes = after_upload.getvalue()
        before_name = before_upload.name
        after_name = after_upload.name
    elif gee_pair is not None:
        bp = Path(gee_pair["before_path"])
        ap = Path(gee_pair["after_path"])
        if bp.is_file() and ap.is_file():
            before_bytes = bp.read_bytes()
            after_bytes = ap.read_bytes()
            before_name = bp.name
            after_name = ap.name

    if before_bytes is None or after_bytes is None:
        st.error("Please upload both Before and After satellite images (or prepare a GEE pair).")
    else:
        with st.spinner("Validating alignment, running tiled inference, and computing metrics..."):
            try:
                res = client.predict_files(
                    before_name=before_name,
                    before_bytes=before_bytes,
                    after_name=after_name,
                    after_bytes=after_bytes,
                    model_name=selected_model,
                    checkpoint_name=checkpoint_name,
                    threshold=threshold,
                    tile_size=int(tile_size),
                    stride=int(stride),
                    normalize_mode=norm_mode,
                    postprocess_enabled=post_enabled,
                    morph_operation=morph_op,
                    min_region_pixels=int(min_region),
                    include_xai=include_xai,
                )
                st.session_state["last_result_id"] = res["result_id"]
                st.success(f"Prediction complete — Result ID: `{res['result_id']}`")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

active_id = st.session_state.get("last_result_id")
if active_id:
    try:
        res = client.get_result(active_id)
        geo = res.get("geospatial", {})
        render_metric_cards(
            [
                {"title": "Change Percentage", "value": f"{res['change_percentage']:.2f}%"},
                {"title": "Changed Pixels", "value": f"{res['changed_pixels']:,}"},
                {
                    "title": "Valid / Masked Pixels",
                    "value": f"{res['total_valid_pixels']:,}",
                    "sub": f"{res['masked_invalid_pixels']:,} masked",
                },
                {
                    "title": "Changed Area" if geo.get("is_georeferenced") else "Inference Time",
                    "value": (
                        f"{geo['changed_area_ha']:.4f} ha"
                        if geo.get("changed_area_ha") is not None
                        else f"{res['inference_time_sec']:.2f} s"
                    ),
                    "sub": geo.get("area_calculation_method", ""),
                },
            ]
        )

        # Display warnings cleanly (e.g., RGB missing cloud/CRS bands)
        warnings = res.get("metadata", {}).get("alignment", {}).get("warnings", [])
        if warnings:
            with st.expander("Preprocessing & Compatibility Notes", expanded=False):
                for w in dict.fromkeys(warnings):
                    st.write(f"- {w}")

        b_png = client.get_artifact_bytes(active_id, "before_png")
        a_png = client.get_artifact_bytes(active_id, "after_png")
        m_png = client.get_artifact_bytes(active_id, "png")
        o_png = client.get_artifact_bytes(active_id, "overlay")

        v1, v2, v3, v4 = st.columns(4)
        with v1:
            if b_png:
                st.image(b_png, caption="Before (T1)", use_container_width=True)
        with v2:
            if a_png:
                st.image(a_png, caption="After (T2)", use_container_width=True)
        with v3:
            if m_png:
                st.image(m_png, caption="Predicted Change Mask", use_container_width=True)
        with v4:
            if o_png:
                st.image(o_png, caption="Change Overlay", use_container_width=True)

        st.markdown("#### Export Prediction Artifacts")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            if m_png:
                st.download_button(
                    "Download Mask PNG",
                    data=m_png,
                    file_name=f"{active_id}_mask.png",
                    mime="image/png",
                    use_container_width=True,
                )
        with d2:
            if o_png:
                st.download_button(
                    "Download Overlay PNG",
                    data=o_png,
                    file_name=f"{active_id}_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )
        with d3:
            csv_b = client.get_artifact_bytes(active_id, "csv")
            if csv_b:
                st.download_button(
                    "Download CSV Metrics",
                    data=csv_b,
                    file_name=f"{active_id}_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with d4:
            pdf_b = client.get_artifact_bytes(active_id, "pdf")
            if pdf_b:
                st.download_button(
                    "Download PDF Report",
                    data=pdf_b,
                    file_name=f"{active_id}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
    except FileNotFoundError:
        st.session_state.pop("last_result_id", None)
    except Exception as exc:
        st.error(f"Could not render result '{active_id}': {exc}")

