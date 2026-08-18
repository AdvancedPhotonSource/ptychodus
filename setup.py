"""Setuptools shim that builds the ptychodus_store frontend before packaging.

The main project configuration lives in pyproject.toml; this file only exists to
register build_py/sdist subclasses that run `tsc` in src/ptychodus_store/ui/ so
release artifacts ship the compiled UI without requiring maintainers to remember
a pre-build step. The ui/dist/ output stays gitignored (see .gitignore and the
`ptychodus_store/ui/` bullet under CLAUDE.md Conventions); it reaches the sdist
through [tool.setuptools.package-data], which is why sdist has to compile it too
-- setuptools finalizes build_py to collect package data but never runs it.

The compile is best-effort by default: building from a release sdist finds
ui/dist/ already present and needs no Node toolchain, and building from a bare
checkout without `tsc` warns and ships without the web UI rather than failing.
Set PTYCHODUS_STORE_REQUIRE_UI_BUILD=1 to make a missing or stale UI fatal --
the documented release command does exactly that.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup  # type: ignore[import-untyped]
from setuptools.command.build_py import build_py  # type: ignore[import-untyped]
from setuptools.command.sdist import sdist  # type: ignore[import-untyped]

UI_DIR = Path(__file__).parent / 'src' / 'ptychodus_store' / 'ui'
DIST_ENTRY = UI_DIR / 'dist' / 'main.js'
REQUIRE_ENV = 'PTYCHODUS_STORE_REQUIRE_UI_BUILD'

INSTALL_HINT = (
    'ptychodus_store frontend build requires `tsc` on PATH.\n'
    '  Install: `npm install -g typescript`, or use nodeenv:\n'
    '    uv tool install nodeenv\n'
    '    nodeenv --node=lts --prebuilt ~/.local/node-lts\n'
    '    export PATH="$HOME/.local/node-lts/bin:$PATH"\n'
    '    npm install -g typescript\n'
)


def _is_ui_current() -> bool:
    if not DIST_ENTRY.is_file():
        return False

    latest_src = max(
        (p.stat().st_mtime for p in (UI_DIR / 'src').rglob('*.ts')),
        default=0.0,
    )
    return DIST_ENTRY.stat().st_mtime >= latest_src


def _build_ui() -> None:
    if not (UI_DIR / 'tsconfig.json').is_file():
        return  # ui subpackage absent; nothing to build

    if _is_ui_current():
        return

    tsc = shutil.which('tsc')

    if tsc is None:
        sys.stderr.write(INSTALL_HINT)

        if os.environ.get(REQUIRE_ENV):
            sys.stderr.write(f'{REQUIRE_ENV} is set; refusing to build without `tsc`.\n')
            raise SystemExit(2)

        if DIST_ENTRY.is_file():
            sys.stderr.write('Using the prebuilt (possibly stale) ui/dist/.\n')
        else:
            sys.stderr.write('Building without the ptychodus-store web UI.\n')

        return

    subprocess.check_call([tsc], cwd=str(UI_DIR))


class BuildFrontendPy(build_py):
    """Compile the UI before build_py, so wheel builds ship ui/dist/."""

    def run(self) -> None:
        _build_ui()
        super().run()


class BuildFrontendSdist(sdist):
    """Compile the UI before the sdist file list is computed, so ui/dist/ ships in the tarball."""

    def run(self) -> None:
        _build_ui()
        super().run()


setup(cmdclass={'build_py': BuildFrontendPy, 'sdist': BuildFrontendSdist})
