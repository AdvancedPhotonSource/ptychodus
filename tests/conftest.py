"""Skip optional-extra test packages when their dependencies are absent.

Also hosts the shared ``qapp`` fixture. It imports PyQt5 inside the fixture body rather
than at module scope so this conftest stays importable on a bare install, where the
``tests/view/`` and ``tests/controller/`` subtrees that use it are dropped anyway.
"""

from __future__ import annotations

import os
from importlib.util import find_spec

import pytest

# Keeps QApplication([]) from SIGABRTing on headless machines (no X11/Wayland display);
# developers with a real display can pre-set the env var to override.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

# tests/ptychodus_store/ needs the "store" extra plus pytest-asyncio. Its own conftest
# imports sqlalchemy eagerly, so the directory has to be dropped before pytest descends
# into it -- pytest.importorskip in a conftest is reported as an error, not a skip.
_STORE_TEST_DEPS = (
    'aiosqlite',
    'fastapi',
    'fastmcp',
    'pydantic_settings',
    'pytest_asyncio',
    'sqlalchemy',
)

collect_ignore = []

if any(find_spec(name) is None for name in _STORE_TEST_DEPS):
    collect_ignore.append('ptychodus_store')

if find_spec('PyQt5') is None:
    collect_ignore.append('view')
    collect_ignore.append('controller')


@pytest.fixture(scope='session')
def qapp():  # type: ignore[no-untyped-def]
    """A single QApplication shared across all Qt tests in the session."""
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()

    if app is None:
        app = QApplication([])

    return app
