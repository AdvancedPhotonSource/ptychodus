"""MCP bearer tokens: the token file, its verifier, and the rule for serving MCP without one."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ptychodus.webapp.app import create_app
from ptychodus.webapp.auth import (
    ClientTokenVerifier,
    is_loopback_host,
    load_client_tokens,
    mint_client_token,
    revoke_client_token,
)
from ptychodus.webapp.config import Settings

_MCP_HEADERS = {
    'content-type': 'application/json',
    'accept': 'application/json, text/event-stream',
}
_INITIALIZE = {
    'jsonrpc': '2.0',
    'id': 1,
    'method': 'initialize',
    'params': {
        'protocolVersion': '2025-03-26',
        'capabilities': {},
        'clientInfo': {'name': 'test', 'version': '0'},
    },
}


def test_a_minted_token_is_recorded_only_as_its_digest(tmp_path: Path) -> None:
    tokens_file = tmp_path / 'tokens.json'
    token = mint_client_token(tokens_file, 'agent')

    assert tokens_file.stat().st_mode & 0o777 == 0o600
    assert token not in tokens_file.read_text()
    assert [t.client_id for t in load_client_tokens(tokens_file)] == ['agent']


async def test_the_verifier_accepts_a_client_s_token_as_that_client(tmp_path: Path) -> None:
    tokens_file = tmp_path / 'tokens.json'
    agent_token = mint_client_token(tokens_file, 'agent')
    outside_token = mint_client_token(tokens_file, 'outside')
    verifier = ClientTokenVerifier(load_client_tokens(tokens_file))

    agent = await verifier.verify_token(agent_token)
    outside = await verifier.verify_token(outside_token)

    assert agent is not None and agent.client_id == 'agent'
    assert outside is not None and outside.client_id == 'outside'
    assert await verifier.verify_token('not-a-token') is None


async def test_minting_again_replaces_the_client_s_token(tmp_path: Path) -> None:
    tokens_file = tmp_path / 'tokens.json'
    old_token = mint_client_token(tokens_file, 'agent')
    new_token = mint_client_token(tokens_file, 'agent')
    verifier = ClientTokenVerifier(load_client_tokens(tokens_file))

    assert len(load_client_tokens(tokens_file)) == 1
    assert await verifier.verify_token(old_token) is None
    assert await verifier.verify_token(new_token) is not None


def test_revoking_removes_only_that_client(tmp_path: Path) -> None:
    tokens_file = tmp_path / 'tokens.json'
    mint_client_token(tokens_file, 'agent')
    mint_client_token(tokens_file, 'outside')

    revoke_client_token(tokens_file, 'agent')

    assert [t.client_id for t in load_client_tokens(tokens_file)] == ['outside']
    with pytest.raises(LookupError):
        revoke_client_token(tokens_file, 'agent')


def test_a_token_file_others_can_read_is_refused(tmp_path: Path) -> None:
    tokens_file = tmp_path / 'tokens.json'
    mint_client_token(tokens_file, 'agent')
    tokens_file.chmod(0o640)

    with pytest.raises(PermissionError, match='insecure permissions'):
        load_client_tokens(tokens_file)


def test_a_token_file_naming_a_client_twice_is_refused(tmp_path: Path) -> None:
    tokens_file = tmp_path / 'tokens.json'
    entry = {'client_id': 'agent', 'token_sha256': '0' * 64}
    tokens_file.write_text(json.dumps([entry, entry]))
    tokens_file.chmod(0o600)

    with pytest.raises(ValueError, match='more than once: agent'):
        load_client_tokens(tokens_file)


@pytest.mark.parametrize(
    ('host', 'expected'),
    [
        ('127.0.0.1', True),
        ('::1', True),
        ('localhost', True),
        ('0.0.0.0', False),
        ('mcp.example', False),
    ],
)
def test_only_the_loopback_host_counts_as_loopback(host: str, expected: bool) -> None:
    assert is_loopback_host(host) is expected


def _settings(tmp_storage_root: Path, **overrides: Any) -> Settings:
    return Settings(
        storage_root=tmp_storage_root,
        database_url='sqlite+aiosqlite:///:memory:',
        auto_reconcile_on_startup=False,
        **overrides,
    )


@pytest.fixture
def agent_token(tmp_path: Path) -> tuple[Path, str]:
    tokens_file = tmp_path / 'tokens.json'
    return tokens_file, mint_client_token(tokens_file, 'agent')


@pytest.fixture
def client_with_tokens(
    tmp_storage_root: Path, agent_token: tuple[Path, str]
) -> Iterator[TestClient]:
    settings = _settings(tmp_storage_root, mcp_tokens_file=agent_token[0], host='0.0.0.0')
    with TestClient(create_app(settings)) as client:
        yield client


def _initialize(client: TestClient, token: str | None) -> int:
    headers = dict(_MCP_HEADERS)
    if token is not None:
        headers['authorization'] = f'Bearer {token}'
    return client.post('/mcp/', headers=headers, json=_INITIALIZE).status_code


def test_mcp_refuses_a_missing_or_unknown_token(client_with_tokens: TestClient) -> None:
    assert _initialize(client_with_tokens, None) == 401
    assert _initialize(client_with_tokens, 'not-a-token') == 401


def test_mcp_serves_a_known_token(
    client_with_tokens: TestClient, agent_token: tuple[Path, str]
) -> None:
    assert _initialize(client_with_tokens, agent_token[1]) == 200


def test_rest_stays_open_when_mcp_requires_tokens(client_with_tokens: TestClient) -> None:
    assert client_with_tokens.get('/api/v1/health').status_code == 200


def test_mcp_without_tokens_is_open_on_loopback(tmp_storage_root: Path) -> None:
    with TestClient(create_app(_settings(tmp_storage_root, host='127.0.0.1'))) as client:
        assert _initialize(client, None) == 200


def test_mcp_without_tokens_is_refused_beyond_loopback(tmp_storage_root: Path) -> None:
    with pytest.raises(RuntimeError, match='PTYCHODUS_WEBAPP_MCP_TOKENS_FILE'):
        create_app(_settings(tmp_storage_root, host='0.0.0.0'))
