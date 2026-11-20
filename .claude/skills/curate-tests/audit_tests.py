"""Static test-suite audits for the curate-tests skill.

Run from the repository root; every mode prints one candidate per line as
``path:line detail`` and exits 0. Candidates are leads for triage, not verdicts.

    python3 -I .claude/skills/curate-tests/audit_tests.py assertions
    python3 -I .claude/skills/curate-tests/audit_tests.py tolerances [--loose 1e-3] [--achieved log.json]
    python3 -I .claude/skills/curate-tests/audit_tests.py test-only
    python3 -I .claude/skills/curate-tests/audit_tests.py gaps
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import pathlib
import re
from collections.abc import Iterator

_CHECK_NAMES = frozenset({'raises', 'warns', 'approx', 'deprecated_call', 'fail', 'skip'})
_WORD = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _called_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, 'id', '')


def _is_check_call(call: ast.Call, checkers: set[str]) -> bool:
    name = _called_name(call)
    return name in _CHECK_NAMES or name.lstrip('_').startswith('assert') or name in checkers


def _functions(tree: ast.AST) -> Iterator[FunctionNode]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _find_checkers(trees: dict[pathlib.Path, ast.Module]) -> set[str]:
    """Names of functions anywhere under tests/ that assert, directly or through each other."""
    functions = [f for tree in trees.values() for f in _functions(tree)]
    checkers: set[str] = set()

    while True:
        found = {
            f.name
            for f in functions
            if f.name not in checkers
            and any(
                isinstance(x, ast.Assert)
                or (isinstance(x, ast.Call) and _is_check_call(x, checkers))
                for x in ast.walk(f)
            )
        }
        if not found:
            return checkers
        checkers |= found


def _is_existence_only(statement: ast.Assert) -> bool:
    """True for ``isinstance(...)``, ``... is not None`` and ``len(...) > 0``."""
    test = statement.test

    if isinstance(test, ast.Call):
        return _called_name(test) == 'isinstance'

    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, right = test.ops[0], test.comparators[0]

        if isinstance(op, ast.IsNot):
            return isinstance(right, ast.Constant) and right.value is None

        return (
            isinstance(op, (ast.Gt, ast.NotEq))
            and isinstance(test.left, ast.Call)
            and _called_name(test.left) == 'len'
            and isinstance(right, ast.Constant)
            and right.value == 0
        )

    return False


def _body_digest(function: FunctionNode) -> str:
    body = function.body

    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # a docstring does not make two bodies different

    return hashlib.md5(''.join(map(ast.dump, body)).encode()).hexdigest()


def audit_assertions() -> None:
    """VACUOUS: no check at all. WEAK: one existence-only assert. DUPLICATE: identical bodies."""
    trees = {p: ast.parse(p.read_text()) for p in sorted(pathlib.Path('tests').rglob('*.py'))}
    checkers = _find_checkers(trees)
    bodies: dict[str, list[str]] = collections.defaultdict(list)

    for path, tree in trees.items():
        if not path.name.startswith('test_'):
            continue

        for function in _functions(tree):
            if not function.name.startswith('test'):
                continue

            where = f'{path}:{function.lineno} {function.name}'
            nodes = list(ast.walk(function))
            asserts = [x for x in nodes if isinstance(x, ast.Assert)]
            helper = any(isinstance(x, ast.Call) and _is_check_call(x, checkers) for x in nodes)

            if not asserts and not helper:
                print(f'VACUOUS   {where}')
            elif len(asserts) == 1 and not helper and _is_existence_only(asserts[0]):
                print(f'WEAK      {where}')

            bodies[_body_digest(function)].append(where)

    for group in bodies.values():
        if len(group) > 1:
            print(f'DUPLICATE {" == ".join(group)}')


def _achieved_at(achieved: dict[str, list[float]], path: pathlib.Path, line: int) -> str:
    """The recorded error of the call that starts on or a few lines above *line*."""
    call_lines = 6

    for start in range(line, max(0, line - call_lines), -1):
        errors = achieved.get(f'{path}:{start}')

        if errors is not None:
            return f'  achieved abs={errors[0]:.3g} rel={errors[1]:.3g}'

    return '  achieved: not recorded'


def audit_tolerances(*, loose: float, achieved_log: pathlib.Path | None) -> None:
    """Tolerances at or above ``loose`` with no comment or docstring on or above their line.

    With ``achieved_log`` (written by ``tolerance_recorder``), each also shows the
    largest error its call site reached.
    """
    pattern = re.compile(r'\b(rtol|atol|rel|abs)\s*=\s*([0-9.]+(?:e-?[0-9]+)?)\b')
    context_lines = 3
    achieved = {} if achieved_log is None else json.loads(achieved_log.read_text())

    for path in sorted(pathlib.Path('tests').rglob('test_*.py')):
        lines = path.read_text().splitlines()

        for index, line in enumerate(lines):
            for key, value in pattern.findall(line):
                if float(value) < loose:
                    continue

                context = lines[max(0, index - context_lines) : index + 1]

                if not any('#' in c or '"""' in c or "'''" in c for c in context):
                    measured = (
                        '' if achieved_log is None else _achieved_at(achieved, path, index + 1)
                    )
                    print(f'{path}:{index + 1} {key}={value}{measured}')


def _count_words(root: str, glob: str) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()

    for path in pathlib.Path(root).rglob(glob):
        counts.update(_WORD.findall(path.read_text(errors='ignore')))

    return counts


def _exported_names(tree: ast.Module) -> set[str]:
    return {
        element.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, 'id', '') == '__all__' for target in node.targets)
        for element in getattr(node.value, 'elts', [])
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    }


def audit_test_only() -> None:
    """Public undecorated src/ definitions whose name appears in no production file but its own."""
    in_production = _count_words('src', '*.py') + _count_words('scripts', '*.py')
    in_production.update(_count_words('src', '*.ts'))
    in_tests = _count_words('tests', '*.py')

    for path in sorted(pathlib.Path('src/ptychodus').rglob('*.py')):
        tree = ast.parse(path.read_text())
        exported = _exported_names(tree)
        owners: list[ast.Module | ast.ClassDef] = [tree]
        owners.extend(node for node in tree.body if isinstance(node, ast.ClassDef))

        for owner in owners:
            for node in owner.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue

                if node.name.startswith('_') or node.decorator_list or node.name in exported:
                    continue

                if in_production[node.name] == 1 and in_tests[node.name] > 0:
                    print(f'{path}:{node.lineno} {node.name}')


def _is_record(node: ast.ClassDef) -> bool:
    """Dataclasses, enums and the like are covered through the functions that return them."""
    record_bases = ('Enum', 'Flag', 'NamedTuple', 'TypedDict', 'BaseModel')
    bases = {getattr(base, 'id', getattr(base, 'attr', '')) for base in node.bases}
    decorators = {ast.unparse(d).split('(')[0] for d in node.decorator_list}
    return bool(decorators & {'dataclass', 'dataclasses.dataclass'}) or any(
        base.endswith(record_bases) for base in bases
    )


def audit_gaps() -> None:
    """Public module-level api/ functions and behavioural classes that no test names."""
    in_tests = _count_words('tests', '*.py')

    for path in sorted(pathlib.Path('src/ptychodus/api').rglob('*.py')):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef) and _is_record(node):
                continue

            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.startswith('_'):
                if in_tests[node.name] == 0:
                    print(f'{path}:{node.lineno} {node.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest='mode', required=True)
    modes.add_parser('assertions', help=audit_assertions.__doc__)
    tolerances = modes.add_parser('tolerances', help=audit_tolerances.__doc__)
    tolerances.add_argument('--loose', type=float, default=1.0e-3)
    tolerances.add_argument('--achieved', type=pathlib.Path, default=None)
    modes.add_parser('test-only', help=audit_test_only.__doc__)
    modes.add_parser('gaps', help=audit_gaps.__doc__)
    args = parser.parse_args()

    if args.mode == 'assertions':
        audit_assertions()
    elif args.mode == 'tolerances':
        audit_tolerances(loose=args.loose, achieved_log=args.achieved)
    elif args.mode == 'test-only':
        audit_test_only()
    else:
        audit_gaps()


if __name__ == '__main__':
    main()
