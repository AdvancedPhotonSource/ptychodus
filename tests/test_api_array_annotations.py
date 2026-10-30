"""Guard that no annotation in ``api/`` leaves a numpy array or dtype unparameterized.

``numpy.ndarray`` means ``ndarray[Any, dtype[Any]]`` and ``numpy.dtype`` means
``dtype[Any]``, so either one annotates a value without constraining it. The layer's
whole job is to state what an array holds, and a bare one states nothing while looking
like it does.

Mypy cannot catch this. ``disallow_any_generics`` fires on a generic whose parameters
were omitted, but numpy's stubs give every parameter of ``ndarray``, ``dtype`` and the
scalar types a PEP 696 default, so nothing is missing from the checker's point of view
and the flag stays silent. That is why the rule is enforced here instead.

A dtype that genuinely is not promised has a name: ``AnyArrayType`` in
:mod:`ptychodus.api.typing`, which says so deliberately rather than by omission.

The scan is an AST walk rather than an import so it holds on a bare install, and it
recurses into subscripts, where a bare name hides most easily -- ``tuple[ndarray, ...]``
and ``ndarray | None`` read as annotated until you look at what is inside.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ptychodus.api

_API_ROOT = Path(ptychodus.api.__file__).parent

# Bound TypeVars and casts are how a shape- and dtype-preserving helper is written, and
# both name the unparameterized class legitimately: `TypeVar('_T', bound=numpy.ndarray)`
# constrains nothing on its own, and a cast target is checked where it is used. Only
# annotations are scanned, so neither is reached.
_BARE = ('ndarray', 'dtype')


def _is_bare_numpy(node: ast.expr) -> str | None:
    """Return the attribute name if `node` is an unsubscripted ``numpy.<name>``."""
    if (
        isinstance(node, ast.Attribute)
        and node.attr in _BARE
        and isinstance(node.value, ast.Name)
        and node.value.id == 'numpy'
    ):
        return node.attr

    return None


def _offenses(path: Path) -> list[str]:
    """Return one message per bare ``numpy.ndarray`` / ``numpy.dtype`` in an annotation."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: list[str] = []

    def walk_annotation(node: ast.expr, line: int) -> None:
        if isinstance(node, ast.Subscript):
            # The base of a subscript is exactly the parameterized form being looked for,
            # so only what is inside the brackets is still worth descending into.
            walk_annotation(node.slice, line)
            return

        name = _is_bare_numpy(node)

        if name is not None:
            found.append(f'{path.name}:{line}: bare numpy.{name}')

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                walk_annotation(child, line)

    for node in ast.walk(tree):
        annotations: list[ast.expr | None] = []

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            annotations = [a.annotation for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
            annotations += [args.vararg.annotation if args.vararg else None]
            annotations += [args.kwarg.annotation if args.kwarg else None]
            annotations.append(node.returns)
        elif isinstance(node, ast.AnnAssign):
            annotations = [node.annotation]

        for annotation in annotations:
            if annotation is not None:
                walk_annotation(annotation, annotation.lineno)

    return found


_MODULES = sorted(_API_ROOT.rglob('*.py'))


def test_modules_were_discovered() -> None:
    """A path change that silently found nothing would make this file vacuous."""
    assert len(_MODULES) >= 20


@pytest.mark.parametrize('path', _MODULES, ids=lambda p: p.name)
def test_no_bare_numpy_annotations(path: Path) -> None:
    offenses = _offenses(path)
    assert not offenses, (
        'Annotate what the array holds, with an alias from ptychodus.api.typing or a '
        'shape-typed alias beside the class that produces it. Where the dtype really is '
        'unknown, say so with AnyArrayType:\n  ' + '\n  '.join(offenses)
    )
