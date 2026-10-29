#!/usr/bin/env python
"""Count lines of Python under each top-level subpackage of ``src/ptychodus``.

Every line of every ``.py`` file is classified exactly once, in this order of precedence:

- **docstring** — inside a module, class or function docstring
- **code** — carries a token other than a comment, including the continuation lines of a
  multi-line string that is not a docstring
- **comment** — carries only a comment
- **blank** — whitespace only

Files directly under ``src/ptychodus`` are reported together on a ``(top level)`` row.

    python scripts/count_lines.py
    python scripts/count_lines.py --package src/ptychodus_store

No packaged dependency — this file is intentionally not shipped with the wheel.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from pathlib import Path
import argparse
import ast
import io
import sys
import tokenize


@dataclass
class LineCounts:
    files: int = 0
    code: int = 0
    docstring: int = 0
    comment: int = 0
    blank: int = 0

    @property
    def total(self) -> int:
        return self.code + self.docstring + self.comment + self.blank

    def __iadd__(self, other: LineCounts) -> LineCounts:
        for field in fields(self):
            setattr(self, field.name, getattr(self, field.name) + getattr(other, field.name))

        return self


def _docstring_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body

            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                expr = body[0]
                lines.update(range(expr.lineno, (expr.end_lineno or expr.lineno) + 1))

    return lines


def count_file(path: Path) -> LineCounts:
    source = path.read_text(encoding='utf-8')
    docstring_lines = _docstring_lines(ast.parse(source, filename=str(path)))
    code_lines: set[int] = set()
    comment_lines: set[int] = set()
    non_code_tokens = {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.ENDMARKER,
        tokenize.ENCODING,
    }

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            comment_lines.add(token.start[0])
        elif token.type not in non_code_tokens:
            code_lines.update(range(token.start[0], token.end[0] + 1))

    counts = LineCounts(files=1)

    for lineno, line in enumerate(source.splitlines(), start=1):
        if lineno in docstring_lines:
            counts.docstring += 1
        elif lineno in code_lines:
            counts.code += 1
        elif lineno in comment_lines:
            counts.comment += 1
        elif line.strip():
            counts.code += 1  # e.g. a backslash continuation carrying no token
        else:
            counts.blank += 1

    return counts


def _python_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.rglob('*.py')):
        if '__pycache__' not in path.parts:
            yield path


def count_package(package: Path) -> dict[str, LineCounts]:
    rows: dict[str, LineCounts] = {}

    for child in sorted(package.iterdir()):
        if child.is_dir() and child.name != '__pycache__':
            counts = LineCounts()

            for path in _python_files(child):
                counts += count_file(path)

            if counts.files > 0:
                rows[f'{child.name}/'] = counts

    top_level = LineCounts()

    for path in sorted(package.glob('*.py')):
        top_level += count_file(path)

    if top_level.files > 0:
        rows['(top level)'] = top_level

    return rows


def _print_table(rows: dict[str, LineCounts]) -> None:
    total = LineCounts()

    for counts in rows.values():
        total += counts

    columns = ('files', 'code', 'docstring', 'comment', 'blank', 'total')
    name_width = max(len('package'), len('TOTAL'), *(len(name) for name in rows))
    widths = [max(len(column), len(f'{getattr(total, column):,}')) for column in columns]

    def format_row(name: str, values: list[str]) -> str:
        cells = (value.rjust(width) for value, width in zip(values, widths))
        return f'{name.ljust(name_width)}  ' + '  '.join(cells)

    header = format_row('package', list(columns))
    rule = '-' * len(header)
    print(header)
    print(rule)

    for name, counts in rows.items():
        print(format_row(name, [f'{getattr(counts, column):,}' for column in columns]))

    print(rule)
    print(format_row('TOTAL', [f'{getattr(total, column):,}' for column in columns]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        '--package',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'src' / 'ptychodus',
        help='package directory whose subpackages are counted (default: src/ptychodus)',
    )
    args = parser.parse_args()
    package: Path = args.package

    if not package.is_dir():
        parser.error(f'not a directory: {package}')

    print(f'{package}\n')
    _print_table(count_package(package))
    return 0


if __name__ == '__main__':
    sys.exit(main())
