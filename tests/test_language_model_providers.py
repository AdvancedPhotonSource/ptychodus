from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import socket

import httpx
import pytest

from ptychodus.api.language_model import (
    LanguageModelProvider,
    LanguageModelProviderError,
    WireProtocol,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.model.agent import AgentSettings, LanguageModelProviderSelection
from ptychodus.model.plugins import PluginChooser
from ptychodus.model.settings import SettingsRegistry
from ptychodus.plugins.language_models import (
    AnthropicLanguageModelProvider,
    ArgoLanguageModelProvider,
    GoogleLanguageModelProvider,
    OpenAICompatibleLanguageModelProvider,
    OpenAILanguageModelProvider,
)
from ptychodus.plugins.language_models.argo import ArgoOptions, get_argo_user, get_gateway_url
from ptychodus.plugins.language_models.openai_compatible import OpenAICompatibleOptions

GATEWAY_URL = 'https://apps.inside.anl.gov/argoapi'
_CREDENTIAL_ENV_VARS = (
    'ANTHROPIC_API_KEY',
    'ARGO_USER',
    'GEMINI_API_KEY',
    'GOOGLE_API_KEY',
    'OPENAI_API_KEY',
    'LOCAL_KEY',
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in _CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('Touched the network.')

    monkeypatch.setattr(socket.socket, 'connect', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)
    monkeypatch.setattr(httpx, 'get', refuse)


class _Recorder:
    """Stands in for ``httpx.get``, answering with *body* and recording the request."""

    def __init__(self, body: Any) -> None:
        self._body = body
        self.url = ''
        self.headers: Mapping[str, str] = {}

    def __call__(self, url: str, *, headers: Mapping[str, str], **kwargs: Any) -> httpx.Response:
        self.url = url
        self.headers = headers
        return httpx.Response(200, json=self._body, request=httpx.Request('GET', url))


def _answer(monkeypatch: pytest.MonkeyPatch, body: Any) -> _Recorder:
    recorder = _Recorder(body)
    monkeypatch.setattr(httpx, 'get', recorder)
    return recorder


# Offline guarantee


def test_registering_and_resolving_touch_no_network(
    monkeypatch: pytest.MonkeyPatch, no_network: None
) -> None:
    monkeypatch.setenv('ARGO_USER', 'mdearing')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'a')
    monkeypatch.setenv('OPENAI_API_KEY', 'o')
    monkeypatch.setenv('GEMINI_API_KEY', 'g')

    registry = PluginRegistry.load_plugins()
    catalog = registry.language_model_providers

    assert {plugin.simple_name for plugin in catalog} == {
        'Anthropic',
        'Argo',
        'Google',
        'OpenAI',
        'OpenAICompatible',
    }

    settings_registry = SettingsRegistry()
    selection = LanguageModelProviderSelection(
        settings_registry, AgentSettings(settings_registry), PluginChooser(catalog)
    )

    for options in selection:
        options.model.set_value('some-model')
        options.resolve_endpoint()


def test_selection_persists_each_provider_in_its_own_group() -> None:
    settings_registry = SettingsRegistry()
    settings = AgentSettings(settings_registry)
    catalog = PluginRegistry.load_plugins().language_model_providers
    selection = LanguageModelProviderSelection(settings_registry, settings, PluginChooser(catalog))

    assert selection.get_current().plugin.simple_name == 'Argo'

    selection.provider.set_value('OpenAI-compatible (local)')

    assert settings.provider.get_value() == 'OpenAICompatible'
    assert settings_registry['AgentArgo'].parameters()['Model'].get_value() == 'claudesonnet55'

    base_url = settings_registry['AgentOpenAICompatible'].parameters()['BaseUrl']
    base_url.set_value('http://10.0.0.5:8000/v1')
    selection.get_current_model().set_value('llama')
    endpoint = selection.resolve_endpoint()

    assert endpoint.base_url == 'http://10.0.0.5:8000/v1'
    assert endpoint.model_id == 'llama'


def test_selection_refuses_empty_model() -> None:
    settings_registry = SettingsRegistry()
    settings = AgentSettings(settings_registry)
    settings.provider.set_value('OpenAI')
    catalog = PluginRegistry.load_plugins().language_model_providers
    selection = LanguageModelProviderSelection(settings_registry, settings, PluginChooser(catalog))

    with pytest.raises(LanguageModelProviderError, match='Choose'):
        selection.resolve_endpoint()


# Argo


@pytest.mark.parametrize(
    'base_url',
    [GATEWAY_URL, f'{GATEWAY_URL}/', f'{GATEWAY_URL}/v1', f'{GATEWAY_URL}/v1/'],
)
def test_gateway_url_strips_api_version(base_url: str) -> None:
    assert get_gateway_url(base_url) == GATEWAY_URL


@pytest.mark.parametrize(
    ('model_id', 'protocol', 'base_url'),
    [
        ('claudesonnet55', WireProtocol.ANTHROPIC_MESSAGES, GATEWAY_URL),
        ('gpt6sol', WireProtocol.OPENAI_RESPONSES, f'{GATEWAY_URL}/v1'),
        ('gemini38flash', WireProtocol.OPENAI_CHAT_COMPLETIONS, f'{GATEWAY_URL}/v1'),
    ],
)
def test_argo_routes_by_vendor(
    monkeypatch: pytest.MonkeyPatch, model_id: str, protocol: WireProtocol, base_url: str
) -> None:
    monkeypatch.setenv('ARGO_USER', 'mdearing')
    options = ArgoOptions(base_url=f'{GATEWAY_URL}/v1')

    endpoint = ArgoLanguageModelProvider().resolve(options, model_id)

    assert endpoint.protocol == protocol
    assert endpoint.base_url == base_url
    assert endpoint.model_id == model_id
    assert endpoint.api_key == 'mdearing'


def test_argo_user_prefers_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ARGO_USER', 'mdearing')
    monkeypatch.setattr('getpass.getuser', lambda: 'login')

    assert get_argo_user(GATEWAY_URL) == 'mdearing'


def test_argo_user_falls_back_to_login_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('getpass.getuser', lambda: 'login')

    assert get_argo_user(GATEWAY_URL) == 'login'


def test_argo_user_allows_loopback_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ARGO_USER', 'mdearing')

    assert get_argo_user('http://localhost:8080/argoapi') == 'mdearing'


def test_argo_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ARGO_USER', 'mdearing')
    recorder = _answer(monkeypatch, {'data': [{'id': 'claudesonnet55'}, {'id': 'gpt6sol'}]})

    models = ArgoLanguageModelProvider().list_models(ArgoOptions())

    assert models == ['claudesonnet55', 'gpt6sol']
    assert recorder.url == f'{GATEWAY_URL}/v1/models'
    assert recorder.headers['Authorization'] == 'Bearer mdearing'


# Credentials and confidentiality


@pytest.mark.parametrize(
    ('provider', 'env_var'),
    [
        (AnthropicLanguageModelProvider(), 'ANTHROPIC_API_KEY'),
        (OpenAILanguageModelProvider(), 'OPENAI_API_KEY'),
        (GoogleLanguageModelProvider(), 'GEMINI_API_KEY'),
    ],
)
def test_missing_credential_names_its_variable(
    provider: LanguageModelProvider, env_var: str
) -> None:
    with pytest.raises(LanguageModelProviderError, match=env_var):
        provider.resolve(provider.get_default_options(), 'some-model')


def test_google_falls_back_to_google_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GOOGLE_API_KEY', 'g')
    provider = GoogleLanguageModelProvider()

    assert provider.resolve(provider.get_default_options(), 'm').api_key == 'g'


@pytest.mark.parametrize(
    ('provider', 'env_var'),
    [
        (ArgoLanguageModelProvider(), 'ARGO_USER'),
        (AnthropicLanguageModelProvider(), 'ANTHROPIC_API_KEY'),
        (OpenAILanguageModelProvider(), 'OPENAI_API_KEY'),
        (GoogleLanguageModelProvider(), 'GEMINI_API_KEY'),
    ],
)
@pytest.mark.parametrize('url', ['http://example.com/api', 'ftp://example.com/api'])
def test_credential_is_never_sent_in_cleartext(
    monkeypatch: pytest.MonkeyPatch, provider: LanguageModelProvider, env_var: str, url: str
) -> None:
    monkeypatch.setenv(env_var, 'secret')
    options = provider.get_default_options().model_copy(update={'base_url': url})

    with pytest.raises(LanguageModelProviderError, match='insecure'):
        provider.resolve(options, 'some-model')

    with pytest.raises(LanguageModelProviderError, match='insecure'):
        provider.list_models(options)


def test_local_server_without_key_accepts_cleartext() -> None:
    options = OpenAICompatibleOptions(base_url='http://10.0.0.5:11434/v1/')

    endpoint = OpenAICompatibleLanguageModelProvider().resolve(options, 'llama')

    assert endpoint.protocol == WireProtocol.OPENAI_CHAT_COMPLETIONS
    assert endpoint.base_url == 'http://10.0.0.5:11434/v1'
    assert endpoint.api_key == ''


def test_local_server_with_key_refuses_cleartext(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOCAL_KEY', 'secret')
    options = OpenAICompatibleOptions(
        base_url='http://10.0.0.5:11434/v1', api_key_env_var='LOCAL_KEY'
    )

    with pytest.raises(LanguageModelProviderError, match='insecure'):
        OpenAICompatibleLanguageModelProvider().resolve(options, 'llama')


def test_local_server_reads_named_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('LOCAL_KEY', 'secret')
    options = OpenAICompatibleOptions(
        base_url='https://llm.beamline.example/v1', api_key_env_var='LOCAL_KEY'
    )

    assert OpenAICompatibleLanguageModelProvider().resolve(options, 'llama').api_key == 'secret'


# Listing


def test_anthropic_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'a')
    recorder = _answer(monkeypatch, {'data': [{'id': 'claude-sonnet-5-5'}], 'has_more': False})
    provider = AnthropicLanguageModelProvider()

    assert provider.list_models(provider.get_default_options()) == ['claude-sonnet-5-5']
    assert recorder.url == 'https://api.anthropic.com/v1/models'
    assert recorder.headers['x-api-key'] == 'a'
    assert 'anthropic-version' in recorder.headers


def test_openai_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'o')
    recorder = _answer(monkeypatch, {'data': [{'id': 'model-a'}, {'id': 'model-b'}]})
    provider = OpenAILanguageModelProvider()

    assert provider.list_models(provider.get_default_options()) == ['model-a', 'model-b']
    assert recorder.url == 'https://api.openai.com/v1/models'
    assert recorder.headers['Authorization'] == 'Bearer o'


def test_google_lists_chat_models_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GEMINI_API_KEY', 'g')
    body = {
        'models': [
            {'name': 'models/gemini-chat', 'supportedGenerationMethods': ['generateContent']},
            {'name': 'models/text-embedding', 'supportedGenerationMethods': ['embedContent']},
        ]
    }
    recorder = _answer(monkeypatch, body)
    provider = GoogleLanguageModelProvider()

    assert provider.list_models(provider.get_default_options()) == ['gemini-chat']
    assert recorder.url == 'https://generativelanguage.googleapis.com/v1beta/models'
    assert recorder.headers['x-goog-api-key'] == 'g'


def test_local_server_lists_models_without_authorization(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _answer(monkeypatch, {'data': [{'id': 'llama'}]})
    provider = OpenAICompatibleLanguageModelProvider()

    assert provider.list_models(provider.get_default_options()) == ['llama']
    assert 'Authorization' not in recorder.headers


def test_unreadable_listing_is_a_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _answer(monkeypatch, {'unexpected': []})
    provider = OpenAICompatibleLanguageModelProvider()

    with pytest.raises(LanguageModelProviderError, match='model list'):
        provider.list_models(provider.get_default_options())
