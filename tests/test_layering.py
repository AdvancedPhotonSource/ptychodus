"""Guard the import boundaries that keep ``api/`` usable on its own.

``ptychodus.api`` is the domain layer: it holds no application state and depends on
nothing else in ptychodus, so a consumer such as ``ptychodus.webapp`` can take it without
taking the GUI session along. Three subpackages must therefore never import
``ptychodus.model``, ``ptychodus.view`` or ``ptychodus.controller``:

- ``ptychodus.api`` itself, whose whole purpose is that boundary;
- ``ptychodus.plugins``, which registers into ``api``'s plugin catalogs and is loaded by
  ``api`` -- a plugin reaching back into ``model`` would make the catalogs depend on the
  layer that builds selection over them;
- ``ptychodus.webapp``, the read-only web service, which reads ``api`` only.

The scan is an AST walk rather than an import so it holds on a bare install, and it
reads every import in a module, including the deferred ones inside a function body.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import ast

import pytest

import ptychodus

_PTYCHODUS_ROOT = Path(ptychodus.__file__).parent


def _modules(root: Path, package: str) -> Iterator[tuple[Path, str]]:
    """Yield each module under *root* with the package its relative imports resolve against.

    That is the containing package for a plain module and the package itself for an
    ``__init__.py``; dropping the last path component gives either.
    """
    for path in sorted(root.rglob('*.py')):
        parts = [package, *path.relative_to(root).with_suffix('').parts]
        yield path, '.'.join(parts[:-1])


def _imported_modules(path: Path, package: str) -> Iterator[tuple[int, str]]:
    """Yield the line and absolute name of every module *path* imports."""
    tree = ast.parse(path.read_text(), filename=str(path))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                base = node.module or ''
            else:
                anchor = package.split('.')
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = '.'.join([*anchor, node.module] if node.module else anchor)

            if node.module is None:
                # `from . import x` names submodules as well as attributes.
                for alias in node.names:
                    yield node.lineno, f'{base}.{alias.name}'
            else:
                yield node.lineno, base


def _violations(root: Path, package: str) -> list[str]:
    forbidden = ('ptychodus.model', 'ptychodus.view', 'ptychodus.controller')
    found: list[str] = []

    for path, containing_package in _modules(root, package):
        for line, module in _imported_modules(path, containing_package):
            if any(module == name or module.startswith(f'{name}.') for name in forbidden):
                found.append(f'{path.relative_to(root.parent)}:{line} imports {module}')

    return found


@pytest.mark.parametrize('subpackage', ['api', 'plugins', 'webapp'])
def test_subpackage_imports_nothing_above_api(subpackage: str) -> None:
    assert _violations(_PTYCHODUS_ROOT / subpackage, f'ptychodus.{subpackage}') == []


def test_relative_imports_that_climb_out_of_api_are_caught(tmp_path: Path) -> None:
    """The scan must resolve relative imports, or ``from ..model`` would slip past it."""
    api = tmp_path / 'api'
    (api / 'sub').mkdir(parents=True)
    (api / 'sub' / 'leaf.py').write_text('from ...model.observer import Observable\n')

    assert _violations(api, 'ptychodus.api') == [
        'api/sub/leaf.py:1 imports ptychodus.model.observer'
    ]
