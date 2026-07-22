"""Setuptools shim that builds the ptychodus_store frontend before packaging.

The main project configuration lives in pyproject.toml; this file only exists to
register a build_py subclass that runs `tsc` in src/ptychodus_store/ui/ so wheel
builds ship the compiled UI without requiring maintainers to remember a
pre-build step. The ui/dist/ output stays gitignored (see .gitignore and
CLAUDE.md Repository Notes).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup  # type: ignore[import-untyped]
from setuptools.command.build_py import build_py  # type: ignore[import-untyped]

UI_DIR = Path(__file__).parent / 'src' / 'ptychodus_store' / 'ui'
DIST_ENTRY = UI_DIR / 'dist' / 'main.js'
SKIP_ENV = 'PTYCHODUS_STORE_SKIP_UI_BUILD'


class BuildFrontend(build_py):
    """Run `tsc` before the standard build_py step so ui/dist/ ships in the wheel."""

    def run(self) -> None:
        self._build_ui()
        super().run()

    def _build_ui(self) -> None:
        if not (UI_DIR / 'tsconfig.json').is_file():
            return  # ui subpackage absent (e.g. sdist without ui/); nothing to build

        if DIST_ENTRY.is_file():
            src_dir = UI_DIR / 'src'
            latest_src = max(
                (p.stat().st_mtime for p in src_dir.rglob('*.ts')),
                default=0.0,
            )
            if DIST_ENTRY.stat().st_mtime >= latest_src:
                return

        tsc = shutil.which('tsc')
        if tsc is None:
            sys.stderr.write(
                'ptychodus_store frontend build requires `tsc` on PATH.\n'
                '  Install: `npm install -g typescript`, or use nodeenv:\n'
                '    uv tool install nodeenv\n'
                '    nodeenv --node=lts --prebuilt ~/.local/node-lts\n'
                '    export PATH="$HOME/.local/node-lts/bin:$PATH"\n'
                '    npm install -g typescript\n'
                f'  To skip the frontend build (ships an empty UI), set {SKIP_ENV}=1.\n'
            )
            if os.environ.get(SKIP_ENV):
                return
            raise SystemExit(2)

        subprocess.check_call([tsc], cwd=str(UI_DIR))


setup(cmdclass={'build_py': BuildFrontend})
