from __future__ import annotations

import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from ptychodus.agent.config import ThinkingLevel
from ptychodus.agent.endpoint import create_chat_model, create_model_settings
from ptychodus.api.language_model import LanguageModelEndpoint, WireProtocol


@pytest.mark.parametrize(
    ('protocol', 'base_url', 'model_type'),
    [
        (WireProtocol.ANTHROPIC_MESSAGES, 'https://api.anthropic.com', AnthropicModel),
        (WireProtocol.OPENAI_RESPONSES, 'https://api.openai.com/v1', OpenAIResponsesModel),
        (WireProtocol.OPENAI_CHAT_COMPLETIONS, 'http://localhost:11434/v1', OpenAIChatModel),
        (WireProtocol.GOOGLE_GEMINI, 'https://generativelanguage.googleapis.com', GoogleModel),
    ],
)
def test_chat_model_speaks_endpoint_protocol(
    protocol: WireProtocol, base_url: str, model_type: type
) -> None:
    model = create_chat_model(LanguageModelEndpoint(protocol, base_url, 'some-model', 'key'))

    assert isinstance(model, model_type)
    assert model.model_name == 'some-model'
    # Clients disagree on whether a base URL keeps a trailing slash.
    assert (model.base_url or '').rstrip('/') == base_url


@pytest.mark.parametrize(
    ('protocol', 'env_var'),
    [
        (WireProtocol.ANTHROPIC_MESSAGES, 'ANTHROPIC_API_KEY'),
        (WireProtocol.OPENAI_CHAT_COMPLETIONS, 'OPENAI_API_KEY'),
        (WireProtocol.GOOGLE_GEMINI, 'GOOGLE_API_KEY'),
    ],
)
def test_chat_model_without_key_ignores_vendor_environment(
    monkeypatch: pytest.MonkeyPatch, protocol: WireProtocol, env_var: str
) -> None:
    monkeypatch.setenv(env_var, 'vendor-secret')
    endpoint = LanguageModelEndpoint(protocol, 'http://localhost:8080/v1', 'some-model')

    model = create_chat_model(endpoint)

    if isinstance(model, GoogleModel):
        # google-genai keeps the key on its inner API client.
        api_key = model.client._api_client.api_key
    else:
        assert isinstance(model, AnthropicModel | OpenAIChatModel)
        client = model.client
        assert isinstance(client, AsyncAnthropic | AsyncOpenAI)
        api_key = client.api_key

    assert api_key == 'api-key-not-set'


def test_endpoint_repr_omits_key() -> None:
    endpoint = LanguageModelEndpoint(
        WireProtocol.OPENAI_RESPONSES, 'https://api.openai.com/v1', 'm', api_key='secret'
    )

    assert 'secret' not in repr(endpoint)


def test_model_settings_default_to_model_choice() -> None:
    assert create_model_settings(ThinkingLevel.DEFAULT, 0) == {}


def test_model_settings_pass_max_tokens() -> None:
    assert create_model_settings(ThinkingLevel.DEFAULT, 4096) == {'max_tokens': 4096}


@pytest.mark.parametrize(
    ('level', 'expected'),
    [
        (ThinkingLevel.OFF, False),
        (ThinkingLevel.LOW, 'low'),
        (ThinkingLevel.MEDIUM, 'medium'),
        (ThinkingLevel.HIGH, 'high'),
    ],
)
def test_model_settings_map_thinking_level(level: ThinkingLevel, expected: bool | str) -> None:
    assert create_model_settings(level, 0) == {'thinking': expected}
