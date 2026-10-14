"""Every packaged console command imports and parses ``--help``.

Cheap insurance for a tree where most commands have no other test: an entry point that
names a module that no longer exists, or a parser that raises while being built, fails
only at the shell otherwise.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

_PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'


def _console_scripts() -> list[tuple[str, str, str]]:
    with _PYPROJECT.open('rb') as handle:
        scripts = tomllib.load(handle)['project']['scripts']

    entries = []
    for command, target in scripts.items():
        module, _, function = target.partition(':')
        entries.append((command, module, function))
    return sorted(entries)


@pytest.mark.parametrize(('command', 'module', 'function'), _console_scripts())
def test_entry_point_imports_and_exposes_its_callable(
    command: str, module: str, function: str
) -> None:
    if module.startswith('ptychodus_store'):
        pytest.importorskip('fastapi')

    if module.startswith('ptychodus.cli.reconstruct_'):
        pytest.importorskip('ptychi')

    imported = importlib.import_module(module)
    assert callable(getattr(imported, function)), f'{command} -> {module}:{function}'


@pytest.mark.parametrize(('command', 'module', 'function'), _console_scripts())
def test_entry_point_prints_help_and_exits_zero(
    command: str, module: str, function: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if module.startswith('ptychodus_store'):
        pytest.importorskip('fastapi')

    if module.startswith('ptychodus.cli.reconstruct_'):
        pytest.importorskip('ptychi')

    if module == 'ptychodus.__main__':
        pytest.skip('builds a QApplication rather than only a parser')

    if module == 'ptychodus.cli.system_check':
        pytest.skip('takes no arguments: it has no parser to exercise')

    imported = importlib.import_module(module)
    monkeypatch.setattr('sys.argv', [command, '--help'])

    with pytest.raises(SystemExit) as excinfo:
        getattr(imported, function)()

    assert excinfo.value.code == 0, command
