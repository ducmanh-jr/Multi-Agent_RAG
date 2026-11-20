"""Pytest configuration ensuring `main/` is on sys.path for all test suites."""

from __future__ import annotations

import os
import sys

_MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _MAIN_DIR not in sys.path:
    sys.path.insert(0, _MAIN_DIR)
