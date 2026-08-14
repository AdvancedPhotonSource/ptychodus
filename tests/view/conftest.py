"""Shared PyQt5 fixtures for widget tests.

The outer `tests/conftest.py` drops this whole subtree when PyQt5 is missing,
so importing Qt at module top-level is safe here. The `QT_QPA_PLATFORM=offscreen`
default below keeps `QApplication([])` from `SIGABRT`ing on headless machines
(no X11/Wayland display); developers with a real display can pre-set the env
var to override.
"""

from __future__ import annotations

import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from PyQt5.QtWidgets import QApplication


@pytest.fixture(scope='session')
def qapp() -> QApplication:
    """A single QApplication shared across all widget tests in the session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
