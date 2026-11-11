"""TerraScope Bi-Temporal Change Detection — 4-Step Guided Intelligence Workflow."""

from __future__ import annotations

import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import (
    get_theme_colors,
    render_metric_cards,
    render_research_note,
    render_step_header,
    setup_page,
)

theme = setup_page(
    "Satellite Change Detection",
    "Identify, isolate, and quantify physical surface changes between two satellite observations using deep vision models.",
    icon="🔍",
)
colors = get_theme_colors(theme)
client = get_app_client()

models_data = client.get_models()
model_catalog = {m["model_id"]: m for m in models_data["models"]}

# ==============================================================================
# STEP 1: SELECT SATELLITE IMAGES
# ==============================================================================
render_step_header(
    1,
    "Select Satellite Imagery",
    "Provide two aligned observations of the exact same location captured at different points in time (T1 Before, T2 After).",
)

# Source Selection
sample_rgb_dir = PROJECT_ROOT / "data" / "samples" / "sample_rgb"
sample_geo_dir = PROJECT_ROOT / "data" / "samples" / "sample_geotiff"
gee_pair = st.session_state.get("gee_prepared_pair")

source_options = ["Upload Custom Images"]
if sample_rgb_dir.is_dir() and (sample_rgb_dir / "before.png").is_file():
    source_options.append("Use Demonstration Sample (RGB)")
if sample_geo_dir.is_dir() and (sample_geo_dir / "before.tif").is_file():
    source_options.append("Use Demonstration Sample (Georeferenced GeoTIFF)")
if gee_pair:
    source_options.append(f"Google Earth Engine Export ({gee_pair.get('task_id', 'Active')})")

source_choice = st.radio(
    "Imagery Source",
    options=source_options,
    horizontal=True,
    label_visibility="collapsed",
)

before_bytes: bytes | None = None
after_bytes: bytes | None = None
before_name = "before.png"
after_name = "after.png"
is_georeferenced_input = False
preview_before = None
preview_after = None

if source_choice == "Upload Custom Images":
    col_u1, col_u2 = st.columns(2)
    with col_u1:
        before_file = st.file_uploader(
            "Before Observation (T1)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="upload_before",
            help="Select the earlier satellite observation.",
        )
        if before_file:
            before_bytes = before_file.getvalue()
            before_name = before_file.name
            try:
                preview_before = Image.open(io.BytesIO(before_bytes))
            except Exception:
                pass
    with col_u2:
        after_file = st.file_uploader(
            "After Observation (T2)",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            key="upload_after",
            help="Select the later satellite observation.",
        )
        if after_file:
            after_bytes = after_file.getvalue()
            after_name = after_file.name
            try:
                preview_after = Image.open(io.BytesIO(after_bytes))
            except Exception:
                pass

elif source_choice == "Use Demonstration Sample (RGB)":
    b_path = sample_rgb_dir / "before.png"
    a_path = sample_rgb_dir / "after.png"
    before_bytes = b_path.read_bytes()
    after_bytes = a_path.read_bytes()
    before_name = "sample_rural_before.png"
    after_name = "sample_urban_after.png"
    preview_before = Image.open(b_path)
    preview_after = Image.open(a_path)
    st.caption("Loaded simulated rural-to-urban transition sample (256x256 px).")

elif source_choice == "Use Demonstration Sample (Georeferenced GeoTIFF)":
    b_path = sample_geo_dir / "before.tif"
    a_path = sample_geo_dir / "after.tif"
    before_bytes = b_path.read_bytes()
    after_bytes = a_path.read_bytes()
    before_name = "sample_sentinel2_t1.tif"
    after_name = "sample_sentinel2_t2.tif"
    is_georeferenced_input = True
    st.caption("Loaded 10m Sentinel-2 GeoTIFF pair with EPSG:32643 projection metadata.")

elif gee_pair:
    b_path = Path(gee_pair["before_path"])
    a_path = Path(gee_pair["after_path"])
    if b_path.is_file() and a_path.is_file():
        before_bytes = b_path.read_bytes()
        after_bytes = a_path.read_bytes()
        before_name = b_path.name
        after_name = a_path.name
        is_georeferenced_input = True
        st.success(f"Loaded prepared Sentinel-2 observation pair from Earth Engine task `{gee_pair.get('task_id')}`.")

# Image Previews & Quality Metadata
if before_bytes and after_bytes:
    prev_col1, prev_col2 = st.columns(2)
    with prev_col1:
        st.markdown(f"**T1 — {before_name}**")
        if preview_before:
            st.image(preview_before, use_container_width=True)
            st.caption(f"Dimensions: {preview_before.width} × {preview_before.height} px | Format: {preview_before.format or 'TIFF/Raster'}")
        else:
            st.info(f"Loaded {before_name} ({len(before_bytes) / 1024:.1f} KB)")
    with prev_col2:
        st.markdown(f"**T2 — {after_name}**")
        if preview_after:
            st.image(preview_after, use_container_width=True)
            st.caption(f"Dimensions: {preview_after.width} × {preview_after.height} px | Format: {preview_after.format or 'TIFF/Raster'}")
        else:
            st.info(f"Loaded {after_name} ({len(after_bytes) / 1024:.1f} KB)")
else:
    st.markdown(
        f"""
        <div style="background: {colors['surface_subtle']}; border: 1px dashed {colors['border']}; border-radius: 6px; padding: 20px; text-align: center; color: {colors['text_muted']}; font-size: 0.9rem;">
            Upload two images above or choose a demonstration sample to begin analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ==============================================================================
# STEP 2: CONFIGURE DETECTION
# ==============================================================================
render_step_header(
    2,
    "Configure Detection Model",
    "Choose an architecture tailored to your spatial scale and target structures.",
)

col_m1, col_m2 = st.columns([1.2, 1.0])

model_descriptions = {
    "siamese_unet": (
        "**Siamese U-Net (CNN)**",
        "Traditional convolutional architecture. Best for small structures, sharp building contours, and fast computation.",
    ),
    "vit": (
        "**Siamese Vision Transformer (ViT)**",
        "Full self-attention across non-overlapping image patches. Best for broad regional context and extensive land-cover transitions.",
    ),
    "swin": (
        "**Siamese Swin Transformer**",
        "Hierarchical shifted-window Transformer. Bridges fine pixel boundaries with broad scene context.",
    ),
}

with col_m1:
    selected_model = st.radio(
        "Architecture",
        options=["siamese_unet", "vit", "swin"],
        format_func=lambda m: model_descriptions[m][0].replace("*", ""),
        horizontal=False,
    )
    st.caption(model_descriptions[selected_model][1])

with col_m2:
    available_ckpts = [
        c["filename"]
        for c in model_catalog.get(selected_model, {}).get("available_checkpoints", [])
    ]
    ckpt_options = ["Backbone Initialization (Untrained Head)"] + available_ckpts
    ckpt_choice = st.selectbox(
        "Model Weights",
        options=ckpt_options,
        help="Select a trained checkpoint if available, or run on backbone initialization for pipeline verification.",
    )
    checkpoint_name = None if ckpt_choice.startswith("Backbone") else ckpt_choice

    if checkpoint_name is None:
        st.markdown(
            f"<span class='badge badge-alert'>BASELINE MODE</span> <span style='font-size:0.78rem; color:{colors['text_muted']};'>Running on backbone weights; fine-tune on LEVIR-CD for calibrated building masks.</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<span class='badge badge-ready'>TRAINED CHECKPOINT</span> <span style='font-size:0.78rem; color:{colors['text_muted']};'>{checkpoint_name}</span>",
            unsafe_allow_html=True,
        )

# Threshold explanation in plain language
st.markdown("#### Detection Sensitivity")
threshold = st.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Determines the minimum model confidence required to flag a pixel as changed.",
)
st.caption(
    "**Sensitivity Guide:** `0.3 – 0.4`: Captures subtle radiometric transitions (may increase noise). `0.5`: Balanced default. `0.6 – 0.8`: Restricts detection strictly to distinct, high-contrast structural changes."
)

# Optional Advanced Technical Settings
with st.expander("Advanced Settings (Tiling, Normalization & Filtering)", expanded=False):
    st.caption("Fine-tune preprocessing and sliding-window blending for large high-resolution scenes.")
    adv_c1, adv_c2, adv_c3 = st.columns(3)
    with adv_c1:
        tile_size = st.selectbox("Tile Size (px)", options=[128, 256, 512], index=1)
        stride = st.selectbox("Tile Stride (px)", options=[96, 128, 192, 256], index=2)
    with adv_c2:
        norm_mode = st.selectbox(
            "Spectral Normalization",
            options=["imagenet", "sentinel2_sr", "percentile", "unit", "zscore"],
            index=1 if is_georeferenced_input else 0,
            help="Choose 'sentinel2_sr' for Surface Reflectance integers (0-10000), 'imagenet' for standard RGB.",
        )
        include_xai = st.checkbox("Generate Explainability Heatmap", value=True)
    with adv_c3:
        post_enabled = st.checkbox("Remove Small Noise Islands", value=False)
        morph_op = st.selectbox(
            "Morphological Filter",
            options=["none", "open", "close", "open_close"],
            disabled=not post_enabled,
        )
        min_region = st.number_input(
            "Min Region Size (px)",
            min_value=0,
            max_value=2000,
            value=16 if post_enabled else 0,
            disabled=not post_enabled,
        )

# ==============================================================================
# STEP 3: RUN DETECTION
# ==============================================================================
render_step_header(
    3,
    "Execute Analysis",
    "Run temporal feature extraction, sliding-window blending, and change probability inference.",
)

can_run = before_bytes is not None and after_bytes is not None
if not can_run:
    st.button("Detect Changes", disabled=True, use_container_width=True)
    st.caption("Please provide both T1 and T2 images in Step 1 to enable detection.")
else:
    if st.button("Detect Changes", type="primary", use_container_width=True):
        with st.spinner("Aligning temporal scenes, processing spatial tiles, and computing change statistics..."):
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
                st.success(f"Analysis completed successfully (Observation ID: `{res['result_id']}`)")
            except Exception as exc:
                st.error(f"Detection failed: {exc}")

# ==============================================================================
# STEP 4: UNDERSTAND RESULTS
# ==============================================================================
active_id = st.session_state.get("last_result_id")
if active_id:
    render_step_header(
        4,
        "Understand & Verify Results",
        "Inspect spatial change masks, quantified coverage statistics, and research-grade overlays.",
    )

    try:
        res = client.get_result(active_id)
        geo = res.get("geospatial", {})
        is_geo = geo.get("is_georeferenced", False)

        # Plain-English Executive Change Summary
        pct = res["change_percentage"]
        st.markdown(
            f"""
            <div style="background: {colors['surface']}; border: 1px solid {colors['border']}; border-left: 4px solid {colors['terracotta']}; border-radius: 6px; padding: 16px 20px; margin-bottom: 20px;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: {colors['text']};">
                    Detected Changes in Approximately {pct:.2f}% of the Monitored Footprint
                </div>
                <div style="font-size: 0.88rem; color: {colors['text_muted']}; margin-top: 4px; line-height: 1.5;">
                    Across {res['total_valid_pixels']:,} valid surface pixels analyzed by <b>{res['model_name']}</b>, {res['changed_pixels']:,} pixels exceeded the {res['threshold']} probability threshold.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quantified Metric Cards
        render_metric_cards(
            [
                {
                    "title": "Changed Area",
                    "value": f"{geo['changed_area_ha']:.3f} ha" if is_geo and geo.get("changed_area_ha") is not None else f"{res['changed_pixels']:,} px",
                    "sub": f"{geo.get('area_calculation_method', 'Image-relative measurement')}",
                },
                {
                    "title": "Change Percentage",
                    "value": f"{pct:.2f}%",
                    "sub": f"{res['changed_pixels']:,} / {res['total_valid_pixels']:,} valid px",
                },
                {
                    "title": "Model Architecture",
                    "value": res["model_name"].upper(),
                    "sub": f"Status: {res.get('training_status', 'baseline')}",
                },
                {
                    "title": "Inference Latency",
                    "value": f"{res['inference_time_sec']:.2f} s",
                    "sub": "Tiled sliding-window execution",
                },
            ]
        )

        # Intuitive Visual Comparison Viewer
        b_png = client.get_artifact_bytes(active_id, "before_png")
        a_png = client.get_artifact_bytes(active_id, "after_png")
        m_png = client.get_artifact_bytes(active_id, "png")
        o_png = client.get_artifact_bytes(active_id, "overlay")

        st.markdown("### Visual Comparison Viewer")
        tab_v1, tab_v2, tab_v3 = st.tabs(["Side-by-Side Comparison", "Change Mask Focus", "Overlay Inspection"])

        with tab_v1:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**Observation T1 (Before)**")
                if b_png:
                    st.image(b_png, use_container_width=True)
            with c2:
                st.markdown("**Observation T2 (After)**")
                if a_png:
                    st.image(a_png, use_container_width=True)
            with c3:
                st.markdown("**Predicted Change Mask**")
                if m_png:
                    st.image(m_png, use_container_width=True)
            with c4:
                st.markdown("**Change Overlay**")
                if o_png:
                    st.image(o_png, use_container_width=True)

        with tab_v2:
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**Observation T2 (Reference Scene)**")
                if a_png:
                    st.image(a_png, use_container_width=True)
            with mc2:
                st.markdown("**Isolated Binary Change Mask (White = Detected Change)**")
                if m_png:
                    st.image(m_png, use_container_width=True)

        with tab_v3:
            st.markdown("**Composite Change Overlay (Orange Highlights Mark Change)**")
            if o_png:
                st.image(o_png, use_container_width=True)

        # Mandatory Scientific Integrity Disclaimer
        render_research_note(
            "Scientific Integrity & Interpretation Guidance",
            "Binary change detection isolates significant optical/radiometric divergence between temporal acquisitions. A positive change prediction does not alone prove a specific land-use transition (such as deforestation, infrastructure construction, or seasonal farming) without multi-class ground truth or complementary spectral indices.",
            is_warning=False,
        )

        # Export Actions
        st.markdown("### Export Artifacts")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        with exp_col1:
            if m_png:
                st.download_button(
                    "Download Change Mask (PNG)",
                    data=m_png,
                    file_name=f"{active_id}_mask.png",
                    mime="image/png",
                    use_container_width=True,
                )
        with exp_col2:
            if o_png:
                st.download_button(
                    "Download Overlay (PNG)",
                    data=o_png,
                    file_name=f"{active_id}_overlay.png",
                    mime="image/png",
                    use_container_width=True,
                )
        with exp_col3:
            csv_b = client.get_artifact_bytes(active_id, "csv")
            if csv_b:
                st.download_button(
                    "Download Metrics (CSV)",
                    data=csv_b,
                    file_name=f"{active_id}_metrics.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        with exp_col4:
            pdf_b = client.get_artifact_bytes(active_id, "pdf")
            if pdf_b:
                st.download_button(
                    "Download Executive Report (PDF)",
                    data=pdf_b,
                    file_name=f"{active_id}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    except FileNotFoundError:
        st.session_state.pop("last_result_id", None)
    except Exception as exc:
        st.error(f"Error rendering analysis results: {exc}")
