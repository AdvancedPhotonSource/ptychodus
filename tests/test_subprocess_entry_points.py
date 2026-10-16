"""Guard that every ``run_subprocess`` entry-point string still resolves.

Entry points are addressed as ``'dotted.module:function'`` strings, and the
child imports them only after it has been spawned. A stale string therefore
survives every import, every type check and every test that does not actually
launch a reconstruction -- it surfaces as a ``ChildError`` wrapping
``ModuleNotFoundError`` on the first real run, typically on a GPU machine and
never in CI.

The strings are discovered rather than enumerated, so a new backend is covered
the day it is added. Both halves are checked without importing the target:
``find_spec`` for the module, and an AST scan of its source for the function.
Importing would pull torch or a backend package and defeat the point of a test
that must pass on a bare install.
"""

from __future__ import annotations

import ast
from importlib.util import find_spec
from pathlib import Path

import pytest

import ptychodus

# The whole package, not just model/: entry points are addressed the same way
# wherever they are written, and a scan narrower than that silently stops
# covering one the day it moves.
_PACKAGE_ROOT = Path(ptychodus.__file__).parent


def _discover_entry_points() -> list[tuple[str, str, str]]:
    """Return (source file, module path, function name) for every entry-point literal.

    An entry point is a module-level assignment whose target name contains
    ``ENTRY`` and whose value is a ``'module:function'`` string literal.
    """
    found: list[tuple[str, str, str]] = []

    for path in sorted(_PACKAGE_ROOT.rglob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))

        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                continue

            names = [t.id for t in targets if isinstance(t, ast.Name)]

            if not any('ENTRY' in name for name in names):
                continue

            value = node.value

            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text = value.value

                if text.count(':') == 1:
                    module, function = text.split(':')
                    found.append((path.name, module, function))

    return found


ENTRY_POINTS = _discover_entry_points()


def test_entry_points_were_discovered() -> None:
    """A regex or AST change that silently finds nothing would make this file vacuous."""
    assert len(ENTRY_POINTS) >= 5


@pytest.mark.parametrize(('source', 'module', 'function'), ENTRY_POINTS)
def test_entry_point_module_resolves(source: str, module: str, function: str) -> None:
    assert find_spec(module) is not None, (
        f'{source} names child module "{module}", which does not exist.'
    )


@pytest.mark.parametrize(('source', 'module', 'function'), ENTRY_POINTS)
def test_entry_point_function_exists(source: str, module: str, function: str) -> None:
    spec = find_spec(module)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(), filename=spec.origin)
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function in defined, (
        f'{source} names child entry "{module}:{function}", '
        f'but that module defines no top-level "{function}".'
    )
