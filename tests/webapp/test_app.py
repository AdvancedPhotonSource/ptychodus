"""The assembled application, served as uvicorn serves it.

The other webapp tests build their own FastAPI app from the routers, or call the MCP
tools on the FastMCP object directly, so none of them passes through ``create_app()``
or its lifespan. That is where the MCP mount lives, and where it broke: the mounted
MCP app's own lifespan never ran, so every request to ``/mcp`` failed with a 500
while every MCP test still passed.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ptychodus.webapp.app import create_app
from ptychodus.webapp.config import Settings

_MCP_HEADERS = {
    'content-type': 'application/json',
    'accept': 'application/json, text/event-stream',
}


@pytest.fixture
def client(tmp_storage_root: Path) -> Iterator[TestClient]:
    settings = Settings(
        storage_root=tmp_storage_root,
        database_url='sqlite+aiosqlite:///:memory:',
        auto_reconcile_on_startup=False,
    )
    # Entering the client runs the application's lifespan, as uvicorn does.
    with TestClient(create_app(settings)) as client:
        yield client


def _rpc_result(body: str) -> Any:
    """The JSON-RPC result, whether the reply came as plain JSON or as an SSE stream."""
    for line in body.splitlines():
        if line.startswith('data:'):
            body = line.removeprefix('data:')
            break

    message = json.loads(body)
    assert 'error' not in message, message
    return message['result']


def _rpc(client: TestClient, method: str, *, session: str | None = None, **params: Any) -> Any:
    headers = dict(_MCP_HEADERS)
    if session is not None:
        headers['mcp-session-id'] = session

    payload = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
    response = client.post('/mcp/', headers=headers, json=payload)
    assert response.status_code == 200, response.text
    return response


def test_mcp_serves_over_http(client: TestClient) -> None:
    """A client can open an MCP session at /mcp and list the tools."""
    response = _rpc(
        client,
        'initialize',
        protocolVersion='2025-03-26',
        capabilities={},
        clientInfo={'name': 'test', 'version': '0'},
    )
    assert _rpc_result(response.text)['serverInfo']['name'] == 'ptychodus-webapp'
    session = response.headers['mcp-session-id']

    client.post(
        '/mcp/',
        headers={**_MCP_HEADERS, 'mcp-session-id': session},
        json={'jsonrpc': '2.0', 'method': 'notifications/initialized'},
    )
    tools = _rpc_result(_rpc(client, 'tools/list', session=session).text)['tools']

    assert {'get_product', 'render_object', 'get_visualization_options'} <= {
        tool['name'] for tool in tools
    }


def test_rest_and_ui_are_served_alongside_mcp(client: TestClient) -> None:
    assert client.get('/api/v1/health').json()['db'] == 'ok'
    assert client.get('/ui/').status_code == 200
