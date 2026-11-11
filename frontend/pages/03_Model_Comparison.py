"""TerraScope — Model Architecture Benchmarking & Comparative Intelligence.

Empirically compares Siamese U-Net, Siamese ViT, and Siamese Swin Transformer
using verified experiment records logged in `experiments/results/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import get_theme_colors, render_research_note, setup_page

setup_page(
    "Architecture Comparison & Benchmarking",
    "Evaluate whether self-attention mechanisms offer measurable advantages over convolutional baselines across identical satellite test pairs.",
    icon="📊",
)

colors = get_theme_colors()
client = get_app_client()
exp_data = client.get_experiments()
comp_rows = exp_data.get("comparison_table", [])
summaries = exp_data.get("experiments", [])

# Architecture Overview in human-understandable terms
st.markdown(
    f"""
    <div style="margin-bottom: 24px;">
        <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
            Neural Architecture Trade-offs in Earth Observation
        </h3>
        <p style="font-size: 0.95rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 16px;">
            Detecting surface transformation between bi-temporal satellite acquisitions requires balancing micro-level boundary precision (individual building walls, narrow road alignments) with macro-level spatial context (agricultural parcel shifts, reservoir depletion).
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

arch_col1, arch_col2, arch_col3 = st.columns(3)

with arch_col1:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 20px; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['sage']}; font-weight: 700; margin-bottom: 6px;">
                Convolutional Baseline
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 8px;">
                Siamese U-Net
            </div>
            <p style="font-size: 0.88rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 12px;">
                Processes bi-temporal branches via shared convolutional kernels. Skip connections preserve localized pixel-level spatial detail.
            </p>
            <div style="font-size: 0.82rem; color: {colors['charcoal']}; background: rgba(0,0,0,0.02); padding: 8px 10px; border-radius: 6px;">
                <strong>Strengths:</strong> Crisp edge resolution, low computational footprint, rapid inference on standard hardware.<br>
                <strong>Limitation:</strong> Receptive field grows gradually; vulnerable to false positives from seasonal surface variance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch_col2:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 20px; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['terracotta']}; font-weight: 700; margin-bottom: 6px;">
                Pure Vision Transformer
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 8px;">
                Siamese ViT
            </div>
            <p style="font-size: 0.88rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 12px;">
                Partitions images into fixed 16×16 spatial patches, evaluating all-to-all global self-attention across the complete scene footprint.
            </p>
            <div style="font-size: 0.82rem; color: {colors['charcoal']}; background: rgba(0,0,0,0.02); padding: 8px 10px; border-radius: 6px;">
                <strong>Strengths:</strong> Immediate global contextual awareness; robust to localized illumination and sensor variations.<br>
                <strong>Limitation:</strong> Patch tokenization can blur fine parcel boundaries; quadratic memory scaling with tile dimension.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with arch_col3:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px solid {colors['border']}; border-radius: 8px; padding: 20px; height: 100%;">
            <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; color: {colors['forest']}; font-weight: 700; margin-bottom: 6px;">
                Hierarchical Hybrid
            </div>
            <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700; color: {colors['forest']}; margin-bottom: 8px;">
                Siamese Swin
            </div>
            <p style="font-size: 0.88rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 12px;">
                Combines multiscale feature pyramids with shifted local window self-attention, bridging CNN resolution and ViT global depth.
            </p>
            <div style="font-size: 0.82rem; color: {colors['charcoal']}; background: rgba(0,0,0,0.02); padding: 8px 10px; border-radius: 6px;">
                <strong>Strengths:</strong> Linear computational complexity; precise multi-resolution segment boundaries for dense structures.<br>
                <strong>Limitation:</strong> Requires larger training datasets to fully converge without overfitting to training distributions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Scientific Integrity Section
render_research_note(
    "Empirical Integrity Protocol: In accordance with scientific research standards, TerraScope does not generate synthetic or estimated benchmark scores. All metrics below are read directly from deterministic evaluation runs saved to experiments/results/. Models that have not yet been trained or evaluated on test partitions are explicitly labeled 'Not evaluated'."
)

st.write("")

# Comparative Benchmark Table
st.markdown(
    f"""
    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 12px;">
        <h4 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 600; color: {colors['forest']}; margin: 0;">
            Architecture Benchmark Registry
        </h4>
        <span style="font-size: 0.82rem; color: {colors['muted']};">Standardized 0.50 decision threshold &bull; LEVIR-CD & OSCD protocols</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if comp_rows:
    comp_df = pd.DataFrame(comp_rows)
    # Reorder and format columns nicely
    cols_order = [
        col for col in [
            "display_name", "architecture_family", "parameters",
            "f1_score", "iou", "precision", "recall", "accuracy",
            "eval_loss", "checkpoint_status"
        ] if col in comp_df.columns
    ]
    formatted_df = comp_df[cols_order].copy() if cols_order else comp_df.copy()
    formatted_df = formatted_df.rename(
        columns={
            "display_name": "Architecture",
            "architecture_family": "Family",
            "parameters": "Parameters",
            "f1_score": "F1-Score",
            "iou": "IoU",
            "precision": "Precision",
            "recall": "Recall",
            "accuracy": "Accuracy",
            "eval_loss": "Test Loss",
            "checkpoint_status": "Weights Status",
        }
    )
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)
else:
    st.info("No benchmark registry records found.")

evaluated_rows = [r for r in comp_rows if r.get("f1_score") not in ("Not evaluated", None)]

if not evaluated_rows:
    st.markdown(
        f"""
        <div style="background-color: {colors['card_bg']}; border: 1px dashed {colors['border']}; border-radius: 8px; padding: 24px; margin-top: 16px;">
            <div style="font-weight: 600; color: {colors['forest']}; font-size: 0.95rem; margin-bottom: 6px;">
                No Empirical Benchmarks Recorded Yet
            </div>
            <div style="font-size: 0.88rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 16px;">
                To generate quantitative metrics across architectures, execute model training and evaluation runs against your target dataset (such as LEVIR-CD building change or OSCD urban growth).
            </div>
            <div style="font-size: 0.82rem; font-weight: 600; color: {colors['charcoal']}; margin-bottom: 6px;">
                Standard Training & Evaluation Commands:
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code(
        "# Train baseline CNN and Transformer models on LEVIR-CD\n"
        "python scripts/train_model.py --model siamese_unet --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 25\n"
        "python scripts/train_model.py --model vit          --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 25\n"
        "python scripts/train_model.py --model swin         --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 25\n\n"
        "# Run controlled test benchmark across all 3 models\n"
        "python scripts/evaluate_model.py --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD",
        language="bash",
    )
else:
    st.markdown("### Segmentation Metrics Comparison")
    melt_records = []
    for r in evaluated_rows:
        for mkey, mlbl in (
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1-Score"),
            ("iou", "IoU"),
            ("accuracy", "Accuracy"),
        ):
            val = r.get(mkey)
            if isinstance(val, (int, float)):
                melt_records.append(
                    {"Model": r.get("display_name", r.get("model_id")), "Metric": mlbl, "Score": float(val)}
                )
    if melt_records:
        mdf = pd.DataFrame(melt_records)
        palette = [colors["forest"], colors["sage"], colors["terracotta"], "#6b8e23", "#2c3e50"]
        fig = px.bar(
            mdf,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1.05],
            color_discrete_sequence=palette,
            template="plotly_white",
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(family="Plus Jakarta Sans, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Logged training curves
    history_runs = [s for s in summaries if isinstance(s.get("history"), list) and s["history"]]
    if history_runs:
        st.markdown("### Empirical Convergence Curves")
        selected_run = st.selectbox(
            "Select Logged Experiment Run",
            options=range(len(history_runs)),
            format_func=lambda idx: history_runs[idx].get("experiment_id", f"Run {idx}"),
        )
        hdf = pd.DataFrame(history_runs[selected_run]["history"])
        c1, c2 = st.columns(2)
        with c1:
            if {"epoch", "train_loss", "val_loss"}.issubset(hdf.columns):
                lfig = px.line(
                    hdf,
                    x="epoch",
                    y=["train_loss", "val_loss"],
                    color_discrete_sequence=[colors["terracotta"], colors["forest"]],
                    title="Loss Convergence (Cross-Entropy + Dice)",
                    template="plotly_white",
                )
                lfig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Plus Jakarta Sans, sans-serif"),
                )
                st.plotly_chart(lfig, use_container_width=True)
        with c2:
            if {"epoch", "val_f1", "val_iou"}.issubset(hdf.columns):
                mfig = px.line(
                    hdf,
                    x="epoch",
                    y=["val_f1", "val_iou"],
                    color_discrete_sequence=[colors["forest"], colors["sage"]],
                    title="Validation F1 & Intersection-over-Union",
                    template="plotly_white",
                )
                mfig.update_layout(
                    margin=dict(l=20, r=20, t=40, b=20),
                    font=dict(family="Plus Jakarta Sans, sans-serif"),
                )
                st.plotly_chart(mfig, use_container_width=True)
