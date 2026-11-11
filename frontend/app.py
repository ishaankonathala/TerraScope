"""Main Streamlit entry point for the Satellite Land-Use Monitoring Dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.overview import render_overview_dashboard

if __name__ == "__main__":
    render_overview_dashboard()
