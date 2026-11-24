"""Every request the chat panel's client makes is one the agent service serves.

`ui/src/chat/agent_api.ts` spells each path and method by hand. This reads them out and
holds them to the agent app's routes, as `tests/webapp/test_ts_route_parity.py` holds
`api.ts` to the webapp's.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.routing import APIRoute
from pydantic_ai.models.test import TestModel

from ptychodus.agent.app import create_app
from ptychodus.agent.config import Settings

_AGENT_API_TS = (
    Path(__file__).parents[2]
    / 'src'
    / 'ptychodus'
    / 'webapp'
    / 'ui'
    / 'src'
    / 'chat'
    / 'agent_api.ts'
)


def _skeleton(path: str) -> str:
    return re.sub(r'\$\{[^}]*\}|\{[^}]*\}', '{}', path)


def _client_requests(source: str) -> set[tuple[str, str]]:
    prefix = re.search(r"const AGENT = '([^']+)';", source)
    assert prefix is not None, 'agent_api.ts names no AGENT prefix'
    requests = {
        ('GET', prefix.group(1) + _skeleton(path))
        for path in re.findall(r'getJson<[^>]*>\(\s*[\'`]([^\'`]+)[\'`]', source)
    }
    requests |= {
        (method, prefix.group(1) + _skeleton(path))
        for method, path in re.findall(r"send\(\s*'(\w+)',\s*[\'`]([^\'`]+)[\'`]", source)
    }
    return requests


def test_every_client_request_is_a_served_route() -> None:
    settings = Settings(_env_file=None, user_header='X-Forwarded-User', mcp_token='t')
    app = create_app(settings, language_model=TestModel())
    served = {
        (method, _skeleton(route.path))
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods or ()
    }

    requests = _client_requests(_AGENT_API_TS.read_text())

    assert len(requests) == 6
    assert requests <= served, requests - served
