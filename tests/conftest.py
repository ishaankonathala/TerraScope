"""Shared pytest configuration and environment safeguards."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repository root is importable regardless of working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Matplotlib uses a writable cache directory in sandboxed environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_cache")

