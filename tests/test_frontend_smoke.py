"""Phase 13 Streamlit multipage dashboard smoke tests using Streamlit AppTest."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PAGES = [
    PROJECT_ROOT / "frontend" / "app.py",
    PROJECT_ROOT / "frontend" / "pages" / "01_Dashboard.py",
    PROJECT_ROOT / "frontend" / "pages" / "02_Change_Detection.py",
    PROJECT_ROOT / "frontend" / "pages" / "03_Model_Comparison.py",
    PROJECT_ROOT / "frontend" / "pages" / "04_Explainable_AI.py",
    PROJECT_ROOT / "frontend" / "pages" / "05_Geospatial_Analysis.py",
    PROJECT_ROOT / "frontend" / "pages" / "06_Dataset_Explorer.py",
    PROJECT_ROOT / "frontend" / "pages" / "07_Satellite_Data_Explorer.py",
    PROJECT_ROOT / "frontend" / "pages" / "08_Reports.py",
]


@pytest.mark.parametrize("page_path", PAGES, ids=lambda p: p.name)
def test_streamlit_page_renders_without_exceptions(page_path: Path) -> None:
    at = AppTest.from_file(str(page_path), default_timeout=20)
    at.run()
    assert not at.exception, f"Page {page_path.name} raised exception: {at.exception}"

