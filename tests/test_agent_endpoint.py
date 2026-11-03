from __future__ import annotations

import pytest
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from ptychodus.model.agent import AgentSettings, ThinkingLevel
from ptychodus.model.agent.endpoint import (
    create_chat_model,
    create_model_settings,
    get_argo_user,
    get_gateway_url,
    get_mcp_server_url,
)
from ptychodus.model.settings import SettingsRegistry

GATEWAY_URL = 'https://apps.inside.anl.gov/argoapi'


@pytest.fixture
def settings() -> AgentSettings:
    return AgentSettings(SettingsRegistry())


@pytest.fixture(autouse=True)
def argo_user(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv('ARGO_USER', 'mdearing')
    return 'mdearing'


@pytest.mark.parametrize(
    'base_url',
    [GATEWAY_URL, f'{GATEWAY_URL}/', f'{GATEWAY_URL}/v1', f'{GATEWAY_URL}/v1/'],
)
def test_gateway_url_strips_api_version(base_url: str) -> None:
    assert get_gateway_url(base_url) == GATEWAY_URL


@pytest.mark.parametrize(
    ('model_name', 'model_type', 'expected_base_url'),
    [
        ('claudesonnet55', AnthropicModel, f'{GATEWAY_URL}/'),
        ('gpt6sol', OpenAIResponsesModel, f'{GATEWAY_URL}/v1/'),
        ('gemini38flash', OpenAIChatModel, f'{GATEWAY_URL}/v1/'),
    ],
)
def test_chat_model_routes_by_vendor(
    model_name: str, model_type: type, expected_base_url: str
) -> None:
    model = create_chat_model(f'{GATEWAY_URL}/v1', model_name)

    assert isinstance(model, model_type)
    assert model.model_name == model_name
    assert model.base_url == expected_base_url


def test_argo_user_prefers_environment(argo_user: str) -> None:
    assert get_argo_user(GATEWAY_URL) == argo_user


def test_argo_user_falls_back_to_login_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ARGO_USER')
    monkeypatch.setattr('getpass.getuser', lambda: 'login')

    assert get_argo_user(GATEWAY_URL) == 'login'


def test_argo_user_allows_loopback_http(argo_user: str) -> None:
    assert get_argo_user('http://localhost:8080/argoapi') == argo_user


@pytest.mark.parametrize('url', ['http://example.com/argoapi', 'ftp://example.com/argoapi'])
def test_argo_user_refuses_insecure_endpoint(url: str) -> None:
    with pytest.raises(ValueError, match='insecure'):
        get_argo_user(url)


def test_chat_model_refuses_insecure_endpoint() -> None:
    with pytest.raises(ValueError, match='insecure'):
        create_chat_model('http://example.com/argoapi', 'claudesonnet55')


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        ('', ''),
        ('http://localhost:8000/mcp', 'http://localhost:8000/mcp'),
        ('https://example.com/mcp', 'https://example.com/mcp'),
        ('http://example.com/mcp', ''),
    ],
)
def test_mcp_server_url_requires_confidential_url(
    settings: AgentSettings, url: str, expected: str
) -> None:
    settings.mcp_server_url.set_value(url)

    assert get_mcp_server_url(settings) == expected


def test_model_settings_default_to_model_choice(settings: AgentSettings) -> None:
    assert create_model_settings(settings) == {}


def test_model_settings_pass_max_tokens(settings: AgentSettings) -> None:
    settings.max_tokens.set_value(4096)

    assert create_model_settings(settings) == {'max_tokens': 4096}


@pytest.mark.parametrize(
    ('level', 'expected'),
    [
        (ThinkingLevel.OFF, False),
        (ThinkingLevel.LOW, 'low'),
        (ThinkingLevel.MEDIUM, 'medium'),
        (ThinkingLevel.HIGH, 'high'),
    ],
)
def test_model_settings_map_thinking_level(
    settings: AgentSettings, level: ThinkingLevel, expected: bool | str
) -> None:
    settings.thinking.set_value(level)

    assert create_model_settings(settings) == {'thinking': expected}


def test_model_settings_ignore_unknown_thinking_level(settings: AgentSettings) -> None:
    settings.thinking.set_value('Ludicrous')

    assert create_model_settings(settings) == {}
