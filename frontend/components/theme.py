"""TerraScope editorial design system, typography, color palette, and layout helpers."""

from __future__ import annotations

import textwrap
from typing import Any
import streamlit as st


def clean_html(html_str: str) -> str:
    """Dedent and strip HTML to prevent Markdown parser from interpreting indentation as code blocks."""
    return textwrap.dedent(html_str).strip()


def render_html(html_str: str) -> None:
    """Safely render HTML in Streamlit without code-block indentation bugs."""
    st.markdown(clean_html(html_str), unsafe_allow_html=True)


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

    /* Native Metric Cards Styled for TerraScope */
    div[data-testid="stMetric"] {{
        background-color: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        min-height: 108px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        overflow: hidden;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {colors['border_focus']};
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {colors['text_muted']} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        margin-bottom: 2px !important;
    }}

    div[data-testid="stMetricLabel"] p {{
        margin: 0 !important;
        font-size: 0.72rem !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        word-break: break-word !important;
    }}

    div[data-testid="stMetricValue"] {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: {colors['text']} !important;
        line-height: 1.2 !important;
        margin-top: 4px !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }}

    div[data-testid="stMetricValue"] div {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: {colors['text']} !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }}

    div[data-testid="stMetricDelta"] {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.76rem !important;
        color: {colors['text_muted']} !important;
        font-weight: 400 !important;
        margin-top: 4px !important;
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }}

    div[data-testid="stMetricDelta"] div {{
        color: {colors['text_muted']} !important;
        font-size: 0.76rem !important;
    }}

    /* Hide the upward/downward delta arrow for neutral subtitles */
    div[data-testid="stMetricDelta"] svg {{
        display: none !important;
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
        sidebar_html = (
            f'<div style="padding: 10px 0 16px 0;">'
            f'<div style="font-family: \'Space Grotesk\', sans-serif; font-size: 1.35rem; font-weight: 700; color: {colors["text"]}; letter-spacing: -0.01em;">TerraScope</div>'
            f'<div style="font-size: 0.76rem; font-weight: 600; color: {colors["terracotta"]}; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 2px;">Satellite Change Intelligence</div>'
            f'<div style="font-size: 0.76rem; color: {colors["text_muted"]}; margin-top: 4px; font-style: italic;">"See how the Earth changes."</div>'
            f'</div>'
        )
        st.markdown(sidebar_html, unsafe_allow_html=True)

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
    hero_html = (
        f'<div class="terrascope-hero">'
        f'<span class="terrascope-eyebrow">TERRASCOPE INTELLIGENCE</span>'
        f'<h1 class="terrascope-title">{title}</h1>'
        f'<p class="terrascope-subtitle">{sub_text}</p>'
        f'</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    return current_theme


def render_step_header(step_number: int, title: str, description: str | None = None) -> None:
    """Render a guided workflow step header."""
    desc_html = f"<div class='step-desc'>{description}</div>" if description else ""
    header_html = (
        f'<div class="workflow-step">'
        f'<span class="step-badge">STEP {step_number}</span>'
        f'<h3 class="step-title">{title}</h3>'
        f'</div>'
        f'{desc_html}'
    )
    st.markdown(header_html, unsafe_allow_html=True)


def render_metric_cards(cards: list[dict[str, Any]]) -> None:
    """Render clean, responsive metric cards using native Streamlit metrics styled with TerraScope theme."""
    cols = st.columns(len(cards), gap="small")
    for col, card in zip(cols, cards):
        with col:
            sub = card.get("sub")
            col.metric(
                label=card["title"],
                value=card["value"],
                delta=sub if sub else None,
                delta_color="off",
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
    note_html = (
        f'<div class="research-note" style="border-left-color: {border_color};">'
        f'<div class="research-note-title" style="color: {title_color};">{note_title}</div>'
        f'<div>{note_text}</div>'
        f'</div>'
    )
    st.markdown(note_html, unsafe_allow_html=True)
