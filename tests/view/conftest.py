"""Shared PyQt5 fixtures for widget tests.

The outer `tests/conftest.py` drops this whole subtree when PyQt5 is missing,
so importing Qt at module top-level is safe here.
"""

from __future__ import annotations

import pytest

from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope='session')
def qapp() -> QApplication:
    """A single QApplication shared across all widget tests in the session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
