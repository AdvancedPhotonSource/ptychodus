"""The agent's configuration: its language model, its MCP connection and its forwarded user."""

from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from ptychodus.agent.config import Settings
from ptychodus.agent.endpoint import resolve_endpoint
from ptychodus.agent.loop import FORWARDED_USER_HEADER, build_agent
from ptychodus.api.language_model import LanguageModelProviderError, WireProtocol
from ptychodus.api.plugins import PluginRegistry
from ptychodus.webapp import mcp_server


def _settings(**fields: Any) -> Settings:
    return Settings(_env_file=None, user_header='X-Forwarded-User', **fields)


def _provider(name: str) -> Any:
    plugin = PluginRegistry.load_plugins().language_model_providers.find_plugin(name)
    assert plugin is not None
    return plugin


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('Touched the network.')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    monkeypatch.setattr(httpx, 'get', refuse)


def test_the_language_model_resolves_from_settings_offline(
    monkeypatch: pytest.MonkeyPatch, no_network: None
) -> None:
    monkeypatch.delenv('LOCAL_KEY', raising=False)
    settings = _settings(
        provider='OpenAICompatible',
        model_id='llama',
        provider_options={'base_url': 'http://10.0.0.5:8000/v1'},
    )

    endpoint = resolve_endpoint(settings, _provider(settings.provider))

    assert endpoint.protocol == WireProtocol.OPENAI_CHAT_COMPLETIONS
    assert endpoint.base_url == 'http://10.0.0.5:8000/v1'
    assert endpoint.model_id == 'llama'


def test_the_provider_s_default_language_model_stands_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ARGO_USER', 'jdoe')

    endpoint = resolve_endpoint(_settings(), _provider('Argo'))

    assert endpoint.model_id == 'claudesonnet55'
    assert endpoint.api_key == 'jdoe'


def test_undeclared_provider_options_are_refused() -> None:
    settings = _settings(provider='OpenAICompatible', model_id='m', provider_options={'bse': 'x'})

    with pytest.raises(LanguageModelProviderError, match='PTYCHODUS_AGENT_PROVIDER_OPTIONS'):
        resolve_endpoint(settings, _provider(settings.provider))


def test_a_provider_without_a_default_needs_a_language_model() -> None:
    with pytest.raises(LanguageModelProviderError, match='PTYCHODUS_AGENT_MODEL_ID'):
        resolve_endpoint(_settings(provider='OpenAI'), _provider('OpenAI'))


def test_the_user_header_must_be_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PTYCHODUS_AGENT_USER_HEADER', raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize(
    'url', ['http://127.0.0.1:8000/mcp/', 'http://localhost:8000/mcp/', 'https://x.org/mcp/']
)
def test_the_mcp_token_goes_to_https_or_loopback(url: str) -> None:
    build_agent(TestModel(), _settings(mcp_url=url, mcp_token='secret'))


def test_the_mcp_token_never_crosses_the_network_in_clear_text() -> None:
    with pytest.raises(ValueError, match='Refusing'):
        build_agent(TestModel(), _settings(mcp_url='http://x.org/mcp/', mcp_token='secret'))


def test_the_forwarded_user_header_is_the_one_the_webapp_logs() -> None:
    assert FORWARDED_USER_HEADER == mcp_server.FORWARDED_USER_HEADER
