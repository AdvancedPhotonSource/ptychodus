"""Fixtures serving the webapp to a real browser, for the opt-in `browser` tests.

The UI is served from the compiled `ui/dist/`, which these fixtures check but never
build: under pytest-xdist every worker would compile into the directory the others
serve from.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Page

from webapp_ui_fixtures import ApiOperations, ServedPage, WebappServer

_UI_DIR = Path(__file__).parents[3] / 'src' / 'ptychodus' / 'webapp' / 'ui'


@pytest.fixture(scope='session')
def ui_build() -> None:
    """Fails unless every `ui/src/**/*.ts` has a compiled `ui/dist/**/*.js` no older."""
    source_dir = _UI_DIR / 'src'
    stale = []

    for source in sorted(source_dir.rglob('*.ts')):
        compiled = (_UI_DIR / 'dist' / source.relative_to(source_dir)).with_suffix('.js')

        if not compiled.is_file() or compiled.stat().st_mtime < source.stat().st_mtime:
            stale.append(str(source.relative_to(_UI_DIR)))

    if stale:
        pytest.fail(f'ui/dist is missing or older than {stale}; run `tsc` in {_UI_DIR}.')


@pytest.fixture(scope='session')
def browser_type_launch_args(browser_type_launch_args: dict[str, Any]) -> dict[str, Any]:
    """Launches the system Chrome where there is one, so no browser need be downloaded.

    An explicit `--browser-channel` wins. Without Chrome, pytest-playwright launches its
    own Chromium, which `uv run playwright install chromium` downloads.
    """
    if 'channel' in browser_type_launch_args or shutil.which('google-chrome') is None:
        return browser_type_launch_args

    return {**browser_type_launch_args, 'channel': 'chrome'}


@pytest.fixture(scope='session')
def api_operations(tmp_path_factory: pytest.TempPathFactory) -> ApiOperations:
    return ApiOperations.from_app(tmp_path_factory.mktemp('openapi'))


@pytest.fixture
def open_ui(
    page: Page, tmp_storage_root: Path, api_operations: ApiOperations, ui_build: None
) -> Iterator[Callable[[], ServedPage]]:
    """Serves the storage root as seeded so far, returning the page on the web UI."""
    server = WebappServer(tmp_storage_root)

    def _open() -> ServedPage:
        return ServedPage(page, server.start(), api_operations)

    yield _open
    server.stop()
