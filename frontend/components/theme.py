"""TerraScope editorial design system, typography, color palette, and layout helpers."""

from __future__ import annotations

from typing import Any
import streamlit as st


def get_theme_colors(theme: str = "dark") -> dict[str, str]:
    if theme == "light":
        c = {
            "bg": "#f8f7f4",
            "surface": "#ffffff",
            "surface_subtle": "#f1efe9",
            "border": "#e2ded6",
            "border_focus": "#1b3b2b",
            "text": "#1a211e",
            "text_muted": "#5a6860",
            "forest": "#1b3b2b",
            "forest_light": "#27543d",
            "sage": "#4e735e",
            "sage_bg": "#edf3ef",
            "terracotta": "#c2562d",
            "terracotta_bg": "#faefe9",
            "sand": "#e8e5dc",
            "sidebar_bg": "#f0eee7",
            "tag_bg": "#e6e3da",
        }
    else:
        c = {
            "bg": "#111614",
            "surface": "#171e1b",
            "surface_subtle": "#1d2622",
            "border": "#27342e",
            "border_focus": "#38a169",
            "text": "#e9eee9",
            "text_muted": "#8d9e93",
            "forest": "#204633",
            "forest_light": "#2e6248",
            "sage": "#659379",
            "sage_bg": "#182b21",
            "terracotta": "#dd6b3f",
            "terracotta_bg": "#2d1b15",
            "sand": "#2c3831",
            "sidebar_bg": "#0d1210",
            "tag_bg": "#202a24",
        }
    # Convenient aliases for flexible page styling
    c["card_bg"] = c["surface"]
    c["muted"] = c["text_muted"]
    c["charcoal"] = c["text"]
    return c


def _generate_css(colors: dict[str, str]) -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    .stApp {{
        background-color: {colors['bg']};
        color: {colors['text']};
    }}

    section[data-testid="stSidebar"] {{
        background-color: {colors['sidebar_bg']};
        border-right: 1px solid {colors['border']};
    }}

    /* Typography */
    h1, h2, h3, h4 {{
        font-family: 'Space Grotesk', -apple-system, sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: {colors['text']};
    }}

    /* TerraScope Brand Header */
    .terrascope-hero {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}

    .terrascope-eyebrow {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: {colors['terracotta']};
        display: inline-block;
        margin-bottom: 6px;
    }}

    .terrascope-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1.25;
        margin: 0 0 6px 0;
        color: {colors['text']};
    }}

    .terrascope-subtitle {{
        font-size: 0.95rem;
        line-height: 1.5;
        color: {colors['text_muted']};
        margin: 0;
    }}

    /* Section Step Headers */
    .workflow-step {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid {colors['border']};
    }}

    .step-badge {{
        font-family: 'Space Grotesk', monospace;
        font-size: 0.75rem;
        font-weight: 700;
        background: {colors['forest']};
        color: #ffffff;
        padding: 3px 9px;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }}

    .step-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin: 0;
        color: {colors['text']};
    }}

    .step-desc {{
        font-size: 0.85rem;
        color: {colors['text_muted']};
        margin: 0 0 12px 0;
    }}

    /* Editorial Metric Cards */
    .stat-strip {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 14px;
        margin: 18px 0;
    }}

    .stat-item {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        padding: 16px 18px;
        transition: border-color 0.15s ease;
    }}

    .stat-item:hover {{
        border-color: {colors['border_focus']};
    }}

    .stat-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {colors['text_muted']};
        margin-bottom: 6px;
    }}

    .stat-value {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: {colors['text']};
        line-height: 1.2;
    }}

    .stat-detail {{
        font-size: 0.76rem;
        color: {colors['text_muted']};
        margin-top: 4px;
    }}

    /* Status Badges */
    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.03em;
    }}

    .badge-ready {{
        background: {colors['sage_bg']};
        color: {colors['sage']};
        border: 1px solid {colors['sage']};
    }}

    .badge-alert {{
        background: {colors['terracotta_bg']};
        color: {colors['terracotta']};
        border: 1px solid {colors['terracotta']};
    }}

    .badge-neutral {{
        background: {colors['tag_bg']};
        color: {colors['text_muted']};
        border: 1px solid {colors['border']};
    }}

    /* Callout & Research Notes */
    .research-note {{
        background: {colors['surface_subtle']};
        border-left: 3px solid {colors['forest_light']};
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        margin: 14px 0;
        font-size: 0.88rem;
        line-height: 1.5;
        color: {colors['text']};
    }}

    .research-note-title {{
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {colors['forest_light']};
        margin-bottom: 4px;
    }}

    /* Buttons */
    .stButton>button {{
        border-radius: 6px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.15s ease;
    }}

    .stButton>button[kind="primary"] {{
        background-color: {colors['forest']};
        color: #ffffff;
        border: 1px solid {colors['forest']};
    }}

    .stButton>button[kind="primary"]:hover {{
        background-color: {colors['forest_light']};
        border-color: {colors['forest_light']};
    }}

    /* Streamlit overrides for visual harmony */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 4px;
        padding: 6px 14px;
        font-weight: 600;
        font-size: 0.88rem;
    }}

    /* Minimalist Image Container */
    .preview-box {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 6px;
        padding: 12px;
        margin-bottom: 12px;
    }}

    .preview-meta {{
        display: flex;
        justify-content: space-between;
        font-size: 0.74rem;
        color: {colors['text_muted']};
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px dashed {colors['border']};
    }}
    </style>
    """


def setup_page(title: str, subtitle: str | None = None, icon: str = "🛰️") -> str:
    """Configure Streamlit page layout and inject the TerraScope editorial theme."""
    st.set_page_config(
        page_title=f"{title} — TerraScope",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "ui_theme" not in st.session_state:
        st.session_state["ui_theme"] = "dark"

    current_theme = st.session_state["ui_theme"]
    colors = get_theme_colors(current_theme)

    # Sidebar Header
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 10px 0 16px 0;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 700; color: {colors['text']}; letter-spacing: -0.01em;">
                    TerraScope
                </div>
                <div style="font-size: 0.76rem; font-weight: 600; color: {colors['terracotta']}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">
                    Satellite Change Intelligence
                </div>
                <div style="font-size: 0.76rem; color: {colors['text_muted']}; margin-top: 4px; font-style: italic;">
                    "See how the Earth changes."
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        theme_mode = st.radio(
            "Interface Mode",
            options=["dark", "light"],
            index=0 if current_theme == "dark" else 1,
            horizontal=True,
            format_func=lambda v: "Forest Dark" if v == "dark" else "Parchment Light",
        )
        if theme_mode != current_theme:
            st.session_state["ui_theme"] = theme_mode
            st.rerun()

        st.divider()

    st.markdown(_generate_css(colors), unsafe_allow_html=True)

    # Hero Banner
    sub_text = subtitle or "Multi-Temporal Earth Observation & Land-Use Change Detection"
    st.markdown(
        f"""
        <div class="terrascope-hero">
            <span class="terrascope-eyebrow">TERRASCOPE INTELLIGENCE</span>
            <h1 class="terrascope-title">{title}</h1>
            <p class="terrascope-subtitle">{sub_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    return current_theme


def render_step_header(step_number: int, title: str, description: str | None = None) -> None:
    """Render a guided workflow step header."""
    theme = st.session_state.get("ui_theme", "dark")
    colors = get_theme_colors(theme)
    desc_html = f"<div class='step-desc'>{description}</div>" if description else ""
    st.markdown(
        f"""
        <div class="workflow-step">
            <span class="step-badge">STEP {step_number}</span>
            <h3 class="step-title">{title}</h3>
        </div>
        {desc_html}
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(cards: list[dict[str, Any]]) -> None:
    """Render clean, editorial metric cards without generic AI boxes."""
    theme = st.session_state.get("ui_theme", "dark")
    colors = get_theme_colors(theme)

    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            sub = f"<div class='stat-detail'>{card.get('sub', '')}</div>" if card.get("sub") else ""
            badge = ""
            if card.get("status") == "ready":
                badge = "<span class='badge badge-ready'>VERIFIED</span>"
            elif card.get("status") == "alert":
                badge = "<span class='badge badge-alert'>ACTION</span>"

            st.markdown(
                f"""
                <div class="stat-item">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="stat-label">{card['title']}</span>
                        {badge}
                    </div>
                    <div class="stat-value">{card['value']}</div>
                    {sub}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_research_note(title_or_text: str, text: str = "", is_warning: bool = False) -> None:
    """Render a clean research note or scientific caveat."""
    theme = st.session_state.get("ui_theme", "dark")
    colors = get_theme_colors(theme)
    if not text:
        note_title = "Scientific Integrity Protocol"
        note_text = title_or_text
    else:
        note_title = title_or_text
        note_text = text

    border_color = colors["terracotta"] if is_warning else colors["forest_light"]
    title_color = colors["terracotta"] if is_warning else colors["sage"]
    st.markdown(
        f"""
        <div class="research-note" style="border-left-color: {border_color};">
            <div class="research-note-title" style="color: {title_color};">{note_title}</div>
            <div>{note_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
