"""Every URL the TypeScript client builds must name a route and only its parameters.

`ui/src/api.ts` spells each REST path and query name by hand, and nothing links them to
the routers. A path that drifts 404s in the browser; a query name that drifts is worse,
since the service ignores a parameter it does not declare and answers 200 with the
default. This reads each URL out of the client's `api` object and holds it to the
webapp's OpenAPI document. `test_ts_client_parity.py` checks the response fields.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from webapp_ui_fixtures import API_PREFIX, ApiOperations

_API_TS = Path(__file__).parents[2] / 'src' / 'ptychodus' / 'webapp' / 'ui' / 'src' / 'api.ts'


def _skip_string(source: str, start: int) -> int:
    """The index past the string or template literal opening at `start`."""
    quote = source[start]
    i = start + 1

    while source[i] != quote:
        if source[i] == '\\':
            i += 2
        elif quote == '`' and source.startswith('${', i):
            i = _skip_code(source, i + 2, '}')
        else:
            i += 1

    return i + 1


def _skip_code(source: str, start: int, close: str) -> int:
    """The index past the `close` ending the code at `start`.

    Literals and comments are skipped whole, so a bracket or quote inside one, like the
    apostrophe in a comment, does not count.
    """
    depth = 0
    i = start

    while True:
        char = source[i]

        if char in '\'"`':
            i = _skip_string(source, i)
            continue

        if source.startswith('//', i):
            i = source.index('\n', i)
            continue

        if source.startswith('/*', i):
            i = source.index('*/', i) + 2
            continue

        if char in '([{':
            depth += 1
        elif char in ')]}':
            if depth == 0 and char == close:
                return i + 1

            depth -= 1

        i += 1


def _skeleton(literal: str) -> str:
    """A string or template literal's text, with each `${...}` replaced by `{}`."""
    out = []
    i = 1

    while i < len(literal) - 1:
        if literal[0] == '`' and literal.startswith('${', i):
            out.append('{}')
            i = _skip_code(literal, i + 2, '}')
        else:
            out.append(literal[i])
            i += 1

    return ''.join(out)


@dataclass(frozen=True)
class _ClientURL:
    """One URL the client builds: its path, and the query names it can send."""

    path: str
    query_names: set[str]


def _render_params(source: str) -> set[str]:
    match = re.search(r'export interface RenderParams \{(.*?)\}', source, re.S)
    assert match is not None, 'api.ts declares no RenderParams'
    return set(re.findall(r'^\s*(\w+)\??:', match.group(1), re.M))


def _braced_body(source: str, opening: str) -> str:
    """The text between the braces of the block whose opening line begins `opening`."""
    start = source.index('{', source.index(opening)) + 1
    return source[start : _skip_code(source, start, '}') - 1]


def _spelled_query_names(code: str) -> set[str]:
    return set(re.findall(r'[?&`](\w+)=(?!=)', code))


def _client_urls(source: str) -> list[_ClientURL]:
    """Every URL the `api` object fetches or links to.

    A fetch is a `get<T>(...)` or `getAll<T>(...)` call. Its query names are each
    `name=` the call spells, the ones `getAll` itself spells for a listing, and, where
    it calls `renderQuery`, every `RenderParams` field and each key of the objects
    passed to it. A link is a template beginning `${API}`.
    """
    body = _braced_body(source, 'export const api = {')
    paging_names = _spelled_query_names(_braced_body(source, 'async function getAll<T>'))
    render_params = _render_params(source)
    urls = []

    for call in re.finditer(r'\b(get|getAll)<.*?>\(', body):
        argument = body[call.end() : _skip_code(body, call.end(), ')') - 1].strip()
        names = _spelled_query_names(argument)

        if call.group(1) == 'getAll':
            names |= paging_names
        render = argument.find('renderQuery(')

        if render >= 0:
            start = render + len('renderQuery(')
            render_argument = argument[start : _skip_code(argument, start, ')') - 1]
            names |= render_params | set(re.findall(r'(\w+)\s*:', render_argument))

        literal = argument[: _skip_string(argument, 0)]
        urls.append(_ClientURL(_skeleton(literal).split('?')[0], names))

    for link in re.finditer(r'`\$\{API\}', body):
        literal = body[link.start() : _skip_string(body, link.start())]
        urls.append(_ClientURL(_skeleton(literal).removeprefix('{}'), set()))

    return urls


@pytest.fixture
def api_ts() -> str:
    if not _API_TS.is_file():
        pytest.skip(f'{_API_TS} is absent; ui/src ships only in a source checkout')

    return _API_TS.read_text()


def test_client_prefix_is_the_served_prefix(api_ts: str, tmp_path: Path) -> None:
    match = re.search(r"^const API = '([^']*)';", api_ts, re.M)
    assert match is not None, 'api.ts declares no API prefix'
    assert match.group(1) == API_PREFIX
    assert ApiOperations.from_app(tmp_path).query_names(f'{API_PREFIX}/health') is not None


def test_every_client_url_is_found(api_ts: str) -> None:
    urls = _client_urls(api_ts)
    body = _braced_body(api_ts, 'export const api = {')
    fetches = len(re.findall(r'\b(?:get|getAll)<', body))
    links = body.count('`${API}')

    assert len(urls) == fetches + links
    assert {'/product', '/product/positions/image'} <= {url.path for url in urls}
    assert {'limit', 'offset'} <= next(url for url in urls if url.path == '/product').query_names


def test_every_client_url_names_a_route_and_its_parameters(api_ts: str, tmp_path: Path) -> None:
    operations = ApiOperations.from_app(tmp_path)
    problems = []

    for url in _client_urls(api_ts):
        declared = operations.query_names(API_PREFIX + url.path)

        if declared is None:
            problems.append(f'{url.path}: no route serves it')
        elif undeclared := sorted(url.query_names - declared):
            problems.append(f'{url.path}: the route declares no {undeclared}')

    assert not problems, '\n'.join(problems)
