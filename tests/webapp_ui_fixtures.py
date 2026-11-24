"""The webapp served on a loopback port, and what the web UI asks of it.

`ApiOperations` matches a request path to the GET operation serving it, for checking the
URLs the UI builds against the routes. `WebappServer` and `ServedPage` drive the UI in a
browser. Playwright is imported only for type checking, so the route checks run without
the dev group.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import uvicorn

from ptychodus.webapp.app import create_app
from ptychodus.webapp.config import Settings

if TYPE_CHECKING:
    from playwright.sync_api import ConsoleMessage, Page, Request, Response

API_PREFIX = '/api/v1'
"""The prefix the webapp serves its REST routes under."""


def webapp_settings(storage_root: Path) -> Settings:
    """Settings serving `storage_root`, fixed against the developer's environment.

    Each field a shell variable or a `.env` file could set to break the UI or reach real
    data is given explicitly, since keyword arguments outrank both.
    """
    return Settings(
        storage_root=storage_root,
        database_url='sqlite+aiosqlite:///:memory:',
        auto_reconcile_on_startup=True,
        mcp_tokens_file=None,
        mcp_allow_unauthenticated=True,
        _env_file=None,
    )


class ApiOperations:
    """The webapp's GET operations, each with the query parameters it declares."""

    def __init__(self, openapi: Mapping[str, Any]) -> None:
        self._operations: list[tuple[list[str], frozenset[str]]] = []

        for path, item in openapi['paths'].items():
            operation = item.get('get')

            if operation is not None:
                names = frozenset(
                    parameter['name']
                    for parameter in operation.get('parameters', [])
                    if parameter['in'] == 'query'
                )
                self._operations.append((path.strip('/').split('/'), names))

    @classmethod
    def from_app(cls, storage_root: Path) -> ApiOperations:
        """The operations of the app `webapp_settings(storage_root)` configures."""
        return cls(create_app(webapp_settings(storage_root)).openapi())

    def query_names(self, path: str) -> frozenset[str] | None:
        """The query parameters of the operation serving `path`, or None if none does.

        A path segment in braces, as in `/product/{}/probe/image`, stands for any value.
        Where several operations match, the one with the most literal segments serves it.
        """
        segments = path.strip('/').split('/')
        best: frozenset[str] | None = None
        best_literals = -1

        for template, names in self._operations:
            if len(template) != len(segments):
                continue

            literals = 0

            for expected, actual in zip(template, segments):
                if expected.startswith('{') or actual.startswith('{'):
                    continue

                if expected != actual:
                    break

                literals += 1
            else:
                if literals > best_literals:
                    best, best_literals = names, literals

        return best


class WebappServer:
    """The webapp serving a storage root on a loopback port, from a thread of its own."""

    def __init__(self, storage_root: Path) -> None:
        self._storage_root = storage_root
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def start(self, *, timeout_s: float = 10.0) -> str:
        """Serves the storage root as it is now and returns the base URL.

        The startup reconcile ingests every resource already seeded, so a caller seeds
        first and finds them all listed. The port is bound before uvicorn starts, so no
        other process can take it in between.
        """
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.bind(('127.0.0.1', 0))
        config = uvicorn.Config(
            create_app(webapp_settings(self._storage_root)), log_level='warning'
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, kwargs={'sockets': [self._socket]}, daemon=True
        )
        self._thread.start()
        deadline = time.monotonic() + timeout_s

        # uvicorn offers no startup callback, only this flag.
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError('The webapp exited during startup.')

            if time.monotonic() > deadline:
                raise TimeoutError(f'The webapp did not start within {timeout_s} s.')

            time.sleep(0.02)

        host, port = self._socket.getsockname()
        return f'http://{host}:{port}'

    def stop(self, *, timeout_s: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True

        if self._thread is not None:
            self._thread.join(timeout_s)

        if self._socket is not None:
            self._socket.close()


class ServedPage:
    """A browser page on the served web UI, recording everything that went wrong on it.

    It records uncaught exceptions, console errors, failed requests, responses of status
    400 or above, and every request to the API whose path no operation serves or whose
    query names a parameter the operation does not declare. The last matters because the
    service ignores an undeclared query parameter, answering 200 with the default value.
    """

    def __init__(
        self, page: Page, base_url: str, operations: ApiOperations, *, timeout_ms: float = 10_000
    ) -> None:
        """`timeout_ms` bounds each page action, so a page that never renders fails fast."""
        page.set_default_timeout(timeout_ms)
        self.page = page
        self.base_url = base_url
        self._operations = operations
        self._errors: list[str] = []
        self._error_statuses: list[tuple[int, str]] = []
        page.on('pageerror', self._on_page_error)
        page.on('console', self._on_console)
        page.on('request', self._on_request)
        page.on('requestfailed', self._on_request_failed)
        page.on('response', self._on_response)

    def goto(self, route: str = '') -> None:
        """Opens the web UI at the hash `route`, such as `#product/<uuid>/probe`."""
        self.page.goto(f'{self.base_url}/ui/{route}')

    def assert_clean(self, *, allow_status: Collection[int] = ()) -> None:
        """Fails on anything recorded, except responses with a status in `allow_status`."""
        problems = self._errors + [
            f'{status} {url}' for status, url in self._error_statuses if status not in allow_status
        ]
        assert not problems, 'The web UI went wrong:\n' + '\n'.join(problems)

    def _on_page_error(self, error: Exception) -> None:
        self._errors.append(f'uncaught: {error}')

    def _on_console(self, message: ConsoleMessage) -> None:
        # Chrome logs every response of status 400 or above as a console error too;
        # the response itself is recorded, with its status, by _on_response.
        if message.type == 'error' and not message.text.startswith('Failed to load resource'):
            self._errors.append(f'console: {message.text}')

    def _on_request(self, request: Request) -> None:
        parts = urlsplit(request.url)

        if not parts.path.startswith(f'{API_PREFIX}/'):
            return

        names = self._operations.query_names(parts.path)

        if names is None:
            self._errors.append(f'no operation serves {request.url}')
            return

        undeclared = sorted(set(parse_qs(parts.query, keep_blank_values=True)) - names)

        if undeclared:
            self._errors.append(f'undeclared query {undeclared} in {request.url}')

    def _on_request_failed(self, request: Request) -> None:
        self._errors.append(f'failed: {request.url} ({request.failure})')

    def _on_response(self, response: Response) -> None:
        if response.status >= 400:
            self._error_statuses.append((response.status, response.url))


def query_of(url: str) -> dict[str, str]:
    """The query parameters of `url`, each with its last value."""
    return {name: values[-1] for name, values in parse_qs(urlsplit(url).query).items()}


def path_of(url: str) -> str:
    return urlsplit(url).path
