"""Modern geospatial analytics theme, sidebar controls, and UI components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def _theme_css(mode: str) -> str:
    if mode == "light":
        bg = "#f5f7fa"
        card_bg = "#ffffff"
        text = "#0f172a"
        subtext = "#475569"
        border = "#e2e8f0"
        accent = "#0284c7"
        sidebar_bg = "#eef2f6"
    else:
        bg = "#0b1220"
        card_bg = "#111b2e"
        text = "#e2e8f0"
        subtext = "#94a3b8"
        border = "#1e293b"
        accent = "#22d3ee"
        sidebar_bg = "#0f172a"

    return f"""
    <style>
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {sidebar_bg};
        border-right: 1px solid {border};
    }}
    .geo-card {{
        background: {card_bg};
        border: 1px solid {border};
        border-left: 4px solid {accent};
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }}
    .geo-card-title {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {subtext};
        margin-bottom: 4px;
        font-weight: 600;
    }}
    .geo-card-value {{
        font-size: 1.55rem;
        font-weight: 700;
        color: {text};
        line-height: 1.2;
    }}
    .geo-card-sub {{
        font-size: 0.78rem;
        color: {subtext};
        margin-top: 4px;
    }}
    .geo-banner {{
        background: {card_bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 18px;
    }}
    .geo-pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid {accent};
        color: {accent};
    }}
    </style>
    """


def setup_page(title: str, subtitle: str | None = None, icon: str = "🛰️") -> str:
    """Configure Streamlit page, inject Dark/Light geospatial CSS, and render sidebar."""
    st.set_page_config(
        page_title=f"{title} | TerraViT Satellite Monitoring",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "dark"

    with st.sidebar:
        st.markdown("### 🛰️ TerraViT Monitor")
        st.caption("Multi-Temporal Land-Use Change Intelligence")
        theme_choice = st.radio(
            "Interface Theme",
            options=["dark", "light"],
            index=0 if st.session_state["ui_theme"] == "dark" else 1,
            horizontal=True,
            format_func=lambda v: "Dark" if v == "dark" else "Light",
        )
        st.session_state["ui_theme"] = theme_choice
        st.divider()

    st.markdown(_theme_css(st.session_state["ui_theme"]), unsafe_allow_html=True)

    header_html = f"<div class='geo-banner'><span class='geo-pill'>GEOSPATIAL AI ANALYTICS</span><h2 style='margin:8px 0 4px 0;'>{title}</h2>"
    if subtitle:
        header_html += f"<p style='margin:0;opacity:0.82;'>{subtitle}</p>"
    header_html += "</div>"
    st.markdown(header_html, unsafe_allow_html=True)
    return st.session_state["ui_theme"]


def render_metric_cards(cards: list[dict[str, Any]]) -> None:
    """Render responsive metric cards across columns."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            sub = f"<div class='geo-card-sub'>{card.get('sub', '')}</div>" if card.get("sub") else ""
            st.markdown(
                f"<div class='geo-card'>"
                f"<div class='geo-card-title'>{card['title']}</div>"
                f"<div class='geo-card-value'>{card['value']}</div>"
                f"{sub}"
                f"</div>",
                unsafe_allow_html=True,
            )

