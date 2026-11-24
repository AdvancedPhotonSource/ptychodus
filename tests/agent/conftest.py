"""The agent service against the real webapp, which serves a storage root holding a product."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models import Model

from ptychodus.agent.app import create_app
from ptychodus.agent.config import Settings
from ptychodus.api.storage import StorageLayout, save_product_resource
from ptychodus.webapp.auth import mint_client_token

from agent_fixtures import USER_HEADER, Webapp, sample_product
from webapp_ui_fixtures import WebappServer


@pytest.fixture(scope='module')
def webapp(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Webapp]:
    """The webapp, requiring MCP tokens, with one minted for the agent."""
    root = tmp_path_factory.mktemp('storage')
    product_uuid = save_product_resource(StorageLayout(root), sample_product())
    tokens_file = root.parent / 'tokens.json'
    token = mint_client_token(tokens_file, 'agent')
    server = WebappServer(root, mcp_tokens_file=tokens_file)
    url = server.start()

    try:
        yield Webapp(f'{url}/mcp/', token, product_uuid)
    finally:
        server.stop()


@pytest.fixture
def agent_settings(webapp: Webapp) -> Callable[..., Settings]:
    """Settings for an agent of the webapp, fixed against the developer's environment."""

    def _settings(**overrides: Any) -> Settings:
        fields: dict[str, Any] = {
            'user_header': USER_HEADER,
            'development_user': None,
            'database_url': 'sqlite+aiosqlite:///:memory:',
            'mcp_url': webapp.mcp_url,
            'mcp_token': webapp.token,
            'user_daily_tokens_limit': None,
            **overrides,
        }
        return Settings(_env_file=None, **fields)

    return _settings


@pytest.fixture
def agent_client(
    agent_settings: Callable[..., Settings],
) -> Iterator[Callable[..., TestClient]]:
    """Opens the agent service on `language_model`; each client is closed afterwards."""
    clients: list[TestClient] = []

    def _open(language_model: Model, **overrides: Any) -> TestClient:
        client = TestClient(create_app(agent_settings(**overrides), language_model=language_model))
        client.__enter__()
        clients.append(client)
        return client

    yield _open

    for client in clients:
        client.__exit__(None, None, None)
