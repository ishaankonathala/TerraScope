"""TerraScope — Dataset Explorer & Earth Observation Benchmark Catalogs.

Surfaces official dataset specifications for LEVIR-CD, OSCD, and custom local datasets
with split volumes, spectral band configurations, and verifiable sample previews.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.client import get_app_client
from frontend.components.theme import get_theme_colors, render_research_note, setup_page
from ml.preprocessing.image_loader import SUPPORTED_IMAGE_EXTENSIONS

setup_page(
    "Dataset Explorer & Benchmark Catalogs",
    "Inspect structured satellite datasets, train/val/test split distributions, spectral bands, and ground-truth masks.",
    icon="🗂️",
)

colors = get_theme_colors()
client = get_app_client()
ds_resp = client.get_datasets()
datasets = ds_resp.get("datasets", [])

# Dataset Catalog Header
st.markdown(
    f"""
    <div style="margin-bottom: 20px;">
        <h3 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 600; color: {colors['forest']}; margin-bottom: 8px;">
            Standardized Earth Observation Benchmark Catalogs
        </h3>
        <p style="font-size: 0.95rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 16px;">
            Rigorous evaluation of bi-temporal change detection requires canonical datasets with verified spatial alignment, ground-truth binary masks, and reproducible splits.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_rows = []
for d in datasets:
    if d["id"] == "levir_cd":
        count_str = ", ".join(f"{k}: {v.get('count', 0)}" for k, v in d.get("splits", {}).items())
    elif d["id"] == "oscd":
        count_str = f"{d.get('city_count', 0)} cities"
    else:
        count_str = f"{d.get('pair_count', 0)} sample pairs"

    status_tag = "Available (Local)" if d["available"] else "Not Downloaded"
    summary_rows.append(
        {
            "Dataset": d["name"],
            "Status": status_tag,
            "Volume": count_str,
            "Spectral Bands": ", ".join(d.get("bands", ["R", "G", "B"])),
            "Core Objective": d["purpose"],
            "Filesystem Path": d["path"],
        }
    )

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.write("")
render_research_note(
    "Dataset Integrity Note: Standard public research datasets (LEVIR-CD, OSCD) are not bundled in repository git history due to multi-gigabyte licensing terms. Download instructions and verified schema loaders are provided below."
)
st.write("")

# Detailed Dataset Explorers
for d in datasets:
    is_avail = bool(d["available"])
    tag_bg = "rgba(27, 59, 43, 0.08)" if is_avail else "rgba(194, 86, 45, 0.08)"
    tag_color = colors["forest"] if is_avail else colors["terracotta"]
    tag_label = "Available Locally" if is_avail else "Download Required"

    with st.expander(f"{d['name']} — {tag_label}", expanded=is_avail):
        st.markdown(
            f"""
            <div style="margin-bottom: 12px;">
                <div style="font-size: 0.88rem; color: {colors['muted']}; line-height: 1.5; margin-bottom: 8px;">
                    {d['purpose']}
                </div>
                <div style="font-size: 0.82rem; color: {colors['charcoal']}; margin-bottom: 4px;">
                    <strong>Local Directory:</strong> <code>{d['path']}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if d.get("official_url"):
            st.markdown(f"**Official Reference & License:** [{d['official_url']}]({d['official_url']})")

        if not is_avail:
            st.markdown(
                f"""
                <div style="background-color: {colors['card_bg']}; border: 1px dashed {colors['border']}; border-radius: 6px; padding: 16px; margin-top: 10px;">
                    <div style="font-size: 0.88rem; color: {colors['muted']}; margin-bottom: 8px;">
                        To acquire and prepare this benchmark dataset, execute the automated downloader or place the extracted archives in <code>{d['path']}</code>:
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if d["id"] == "levir_cd":
                st.code(
                    "python scripts/download_data.py --dataset LEVIR-CD --extract-dir data/raw/LEVIR-CD",
                    language="bash",
                )
            elif d["id"] == "oscd":
                st.code(
                    "python scripts/download_data.py --dataset OSCD --extract-dir data/raw/OSCD",
                    language="bash",
                )
        else:
            if d["id"] == "levir_cd":
                sp = st.selectbox("Select Benchmark Split", options=["train", "val", "test"])
                sp_dir = PROJECT_ROOT / "data" / "raw" / "LEVIR-CD" / sp
                files = (
                    sorted(
                        p.name
                        for p in (sp_dir / "A").iterdir()
                        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                    )
                    if (sp_dir / "A").is_dir()
                    else []
                )
                st.write(f"Split `{sp}` contains **{len(files)}** registered image pairs (1024×1024 resolution, 0.5m GSD).")
                if files:
                    chosen = st.selectbox("Select Sample Pair to Preview", options=files[:50])
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.image(Image.open(sp_dir / "A" / chosen), caption=f"Before Image (A) — {chosen}", use_container_width=True)
                    p_col2.image(Image.open(sp_dir / "B" / chosen), caption=f"After Image (B) — {chosen}", use_container_width=True)
                    label_path = sp_dir / "label" / chosen
                    if label_path.exists():
                        p_col3.image(Image.open(label_path), caption=f"Ground-Truth Mask — {chosen}", use_container_width=True)
            elif d["id"] == "oscd":
                st.write(f"Discovered {d.get('city_count', 0)} urban observation footprints:")
                st.write(", ".join(d.get("cities", [])))
            elif d["id"] == "custom_samples":
                st.write(f"Bundled demonstration pairs: **{d.get('pair_count', 0)}** pairs located at `{d['path']}`.")
                pairs = d.get("pairs", [])
                if pairs:
                    chosen_pair = st.selectbox("Select Demonstration Sample", options=[p["name"] for p in pairs])
                    pair_obj = next((p for p in pairs if p["name"] == chosen_pair), None)
                    if pair_obj:
                        c1, c2 = st.columns(2)
                        c1.image(Image.open(pair_obj["before"]), caption="Before Image", use_container_width=True)
                        c2.image(Image.open(pair_obj["after"]), caption="After Image", use_container_width=True)
