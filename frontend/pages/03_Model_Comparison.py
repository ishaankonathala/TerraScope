"""Model Comparison page displaying real experimental benchmark records."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import setup_page

setup_page(
    "Architecture Benchmarking & Comparison",
    "Quantitative comparison of Siamese U-Net, Vision Transformer (ViT), and Swin Transformer from actual experiment logs.",
    icon="📊",
)

client = get_app_client()
exp_data = client.get_experiments()
comp_rows = exp_data["comparison_table"]
summaries = exp_data["experiments"]

comp_df = pd.DataFrame(comp_rows)
st.markdown("### Comparative Benchmark Table")
st.dataframe(comp_df, use_container_width=True, hide_index=True)

evaluated_rows = [r for r in comp_rows if r.get("f1_score") != "Not evaluated"]
if not evaluated_rows:
    st.info(
        "No real dataset training or benchmark evaluations are logged in `experiments/results/` yet. "
        "Metrics strictly display `Not evaluated` until models are trained and benchmarked."
    )
    st.code(
        "python scripts/train_model.py --model siamese_unet --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 20\n"
        "python scripts/train_model.py --model vit --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 20\n"
        "python scripts/train_model.py --model swin --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD --epochs 20\n"
        "python scripts/evaluate_model.py --dataset LEVIR-CD --data-dir data/raw/LEVIR-CD",
        language="bash",
    )
else:
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
                    {"Model": r["display_name"], "Metric": mlbl, "Score": float(val)}
                )
    if melt_records:
        mdf = pd.DataFrame(melt_records)
        fig = px.bar(
            mdf,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1.05],
            title="Segmentation Metric Comparison Across Architectures",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Training curves if logged
    history_runs = [s for s in summaries if isinstance(s.get("history"), list) and s["history"]]
    if history_runs:
        st.markdown("### Logged Training Curves")
        selected_run = st.selectbox(
            "Select Experiment Run",
            options=range(len(history_runs)),
            format_func=lambda idx: history_runs[idx].get("experiment_id", f"Run {idx}"),
        )
        hdf = pd.DataFrame(history_runs[selected_run]["history"])
        c1, c2 = st.columns(2)
        with c1:
            if {"epoch", "train_loss", "val_loss"}.issubset(hdf.columns):
                st.plotly_chart(
                    px.line(
                        hdf,
                        x="epoch",
                        y=["train_loss", "val_loss"],
                        title="Loss Progression",
                    ),
                    use_container_width=True,
                )
        with c2:
            if {"epoch", "val_f1", "val_iou"}.issubset(hdf.columns):
                st.plotly_chart(
                    px.line(
                        hdf,
                        x="epoch",
                        y=["val_f1", "val_iou"],
                        title="Validation F1 & IoU",
                    ),
                    use_container_width=True,
                )

