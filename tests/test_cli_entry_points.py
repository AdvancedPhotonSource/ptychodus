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


# Reconstruct with whichever library an options document names, so they must load on an
# install that has none.
_ANY_LIBRARY_MODULES = frozenset(
    {'ptychodus.cli.reconstruct_prepared', 'ptychodus.cli.reconstruct_subprocess'}
)


@pytest.mark.parametrize(('command', 'module', 'function'), _console_scripts())
def test_entry_point_imports_and_exposes_its_callable(
    command: str, module: str, function: str
) -> None:
    # Each service's cli.py imports its pydantic-settings configuration at module scope.
    if module.startswith(('ptychodus.webapp', 'ptychodus.agent')):
        pytest.importorskip('pydantic_settings')

    if module.startswith('ptychodus.cli.reconstruct_') and module not in _ANY_LIBRARY_MODULES:
        pytest.importorskip('ptychi')

    if module == 'ptychodus.cli.stream':
        pytest.importorskip('pvapy')

    imported = importlib.import_module(module)
    assert callable(getattr(imported, function)), f'{command} -> {module}:{function}'


@pytest.mark.parametrize(('command', 'module', 'function'), _console_scripts())
def test_entry_point_prints_help_and_exits_zero(
    command: str, module: str, function: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each service's cli.py imports its pydantic-settings configuration at module scope.
    if module.startswith(('ptychodus.webapp', 'ptychodus.agent')):
        pytest.importorskip('pydantic_settings')

    if module.startswith('ptychodus.cli.reconstruct_') and module not in _ANY_LIBRARY_MODULES:
        pytest.importorskip('ptychi')

    if module == 'ptychodus.cli.stream':
        pytest.importorskip('pvapy')

    if module == 'ptychodus.__main__':
        pytest.skip('builds a QApplication rather than only a parser')

    if module == 'ptychodus.cli.system_check':
        pytest.skip('takes no arguments: it has no parser to exercise')

    imported = importlib.import_module(module)
    monkeypatch.setattr('sys.argv', [command, '--help'])

    with pytest.raises(SystemExit) as excinfo:
        getattr(imported, function)()

    assert excinfo.value.code == 0, command
