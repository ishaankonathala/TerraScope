"""Tests ensuring no raw HTML leaks onto the dashboard and that metrics render natively."""

from __future__ import annotations

from pathlib import Path
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_dashboard_overview_renders_native_metrics_and_no_raw_html() -> None:
    dashboard_path = PROJECT_ROOT / "frontend" / "pages" / "01_Dashboard.py"
    at = AppTest.from_file(str(dashboard_path), default_timeout=20)
    at.run()

    # 1. Ensure zero exceptions occurred
    assert not at.exception, f"Dashboard threw unexpected exceptions: {at.exception}"

    # 2. Ensure exactly 4 native st.metric widgets rendered
    assert len(at.metric) >= 4, f"Expected at least 4 metric widgets, found {len(at.metric)}"

    metric_labels = [m.label for m in at.metric]
    assert "Analyzed Pairs" in metric_labels
    assert "Detected Change" in metric_labels
    assert "Neural Models" in metric_labels
    assert "Compute Device" in metric_labels

    # 3. Check every single markdown block to guarantee no raw HTML leaked
    forbidden_snippets = [
        '<div class="stat-value">',
        "<div class='stat-value'>",
        '<div class="stat-detail">',
        "<div class='stat-detail'>",
        '<div class="stat-item">',
    ]

    for md in at.markdown:
        md_text = md.value or ""
        for snippet in forbidden_snippets:
            assert snippet not in md_text, f"Found raw HTML snippet '{snippet}' in markdown: {md_text}"

