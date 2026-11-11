"""Dataset Explorer page displaying real dataset splits, bands, and sample imagery."""

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
from frontend.components.theme import setup_page
from ml.preprocessing.image_loader import SUPPORTED_IMAGE_EXTENSIONS

setup_page(
    "Dataset Explorer",
    "Inspect LEVIR-CD, OSCD, and custom sample datasets without fabricating missing benchmarks.",
    icon="🗂️",
)

client = get_app_client()
ds_resp = client.get_datasets()
datasets = ds_resp.get("datasets", [])

summary_rows = []
for d in datasets:
    if d["id"] == "levir_cd":
        count_str = ", ".join(f"{k}: {v['count']}" for k, v in d["splits"].items())
    elif d["id"] == "oscd":
        count_str = f"{d['city_count']} cities"
    else:
        count_str = f"{d['pair_count']} sample pairs"
    summary_rows.append(
        {
            "Dataset": d["name"],
            "Status": "Available" if d["available"] else "Not Installed",
            "Volume": count_str,
            "Bands": ", ".join(d.get("bands", ["R", "G", "B"])),
            "Purpose": d["purpose"],
            "Location": d["path"],
        }
    )

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

for d in datasets:
    with st.expander(f"{d['name']} — Details & Acquisition", expanded=d["available"]):
        st.write(f"**Path:** `{d['path']}`")
        if d.get("official_url"):
            st.write(f"**Official Website & Licensing:** {d['official_url']}")

        if not d["available"]:
            st.warning(
                f"`{d['name']}` is not yet present locally. Public datasets are not bundled "
                "or fabricated; place downloaded files in the path above or run `python scripts/download_data.py`."
            )
        elif d["id"] == "levir_cd":
            sp = st.selectbox("Select LEVIR-CD Split", options=["train", "val", "test"])
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
            st.write(f"Split `{sp}` contains **{len(files)}** image pairs.")
            if files:
                chosen = st.selectbox("Preview Sample", options=files[:50])
                c1, c2, c3 = st.columns(3)
                c1.image(Image.open(sp_dir / "A" / chosen), caption="Before (A)", use_container_width=True)
                c2.image(Image.open(sp_dir / "B" / chosen), caption="After (B)", use_container_width=True)
                c3.image(
                    Image.open(sp_dir / "label" / chosen),
                    caption="Ground-Truth Mask",
                    use_container_width=True,
                )
        elif d["id"] == "oscd":
            st.write(f"Discovered OSCD cities ({d['city_count']}): {', '.join(d['cities'])}")
        elif d["id"] == "custom_samples":
            st.write(f"Available sample pairs: {[p['name'] for p in d['pairs']]}")

