"""Argo Gateway access for the settings-controlled chat endpoint.

Argo serves three native APIs under one gateway root: Anthropic Messages
(``/v1/messages``), OpenAI Responses (``/v1/responses``) and OpenAI-compatible Chat
Completions (``/v1/chat/completions``). Each takes the caller's ANL domain username
as its bearer token in place of an API key.
"""

from urllib.parse import urlparse
import getpass
import logging
import os

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from .settings import AgentSettings, ThinkingLevel

logger = logging.getLogger(__name__)

__all__ = [
    'create_chat_model',
    'create_model_settings',
    'get_argo_user',
    'get_gateway_url',
    'get_mcp_server_url',
]


def _is_confidential_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.scheme == 'https':
        return True

    return parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1', '::1'}


def get_gateway_url(base_url: str) -> str:
    """Return the gateway root, tolerating a trailing ``/`` or ``/v1``."""
    return base_url.rstrip('/').removesuffix('/v1')


def get_argo_user(url: str) -> str:
    """Return the ANL domain username sent to ``url`` as its bearer token.

    Reads ``ARGO_USER``, falling back to the login name. The value is a username (or an
    ``svc``-prefixed service account), never an email address. It is not a secret, but
    it attributes usage and cost, so it is never stored in ``settings.ini``.

    ``Agent/BaseURL`` is a plain string in ``settings.ini``, and batch mode loads that
    file from its input directory. Refusing cleartext endpoints keeps a settings file
    that travels with a dataset from sending the user's identity elsewhere.

    Raises:
        ValueError: if ``url`` is neither ``https`` nor loopback ``http``, or no
            username can be determined.
    """
    if not _is_confidential_url(url):
        raise ValueError(f'Refusing insecure chat endpoint "{url}"; use an https:// URL.')

    user = os.environ.get('ARGO_USER', '')

    if not user:
        try:
            user = getpass.getuser()
        except OSError as exc:
            raise ValueError('Set ARGO_USER to your ANL domain username.') from exc

    return user


def get_mcp_server_url(settings: AgentSettings) -> str:
    """Return the MCP server URL, or an empty string when unset or insecure."""
    url = settings.mcp_server_url.get_value().strip()

    if url and not _is_confidential_url(url):
        logger.warning(f'Ignoring insecure MCP server "{url}"; use an https:// URL.')
        return ''

    return url


def create_chat_model(base_url: str, model_name: str) -> Model:
    """Route ``model_name`` to its vendor-native Argo API.

    ``claude*`` models use Anthropic Messages, ``gpt*`` models OpenAI Responses, and
    every other model OpenAI-compatible Chat Completions.
    """
    gateway_url = get_gateway_url(base_url)
    user = get_argo_user(gateway_url)

    if model_name.startswith('claude'):
        return AnthropicModel(
            model_name, provider=AnthropicProvider(base_url=gateway_url, api_key=user)
        )

    provider = OpenAIProvider(base_url=f'{gateway_url}/v1', api_key=user)

    if model_name.startswith('gpt'):
        return OpenAIResponsesModel(model_name, provider=provider)

    return OpenAIChatModel(model_name, provider=provider)


def create_model_settings(settings: AgentSettings) -> ModelSettings:
    """Return generation settings, leaving out whatever the model should choose itself.

    No sampling parameters are sent: current Argo models variously drop, reject or pin
    ``temperature`` and ``top_p``.
    """
    model_settings = ModelSettings()
    max_tokens = settings.max_tokens.get_value()

    if max_tokens > 0:
        model_settings['max_tokens'] = max_tokens

    thinking = settings.thinking.get_value()

    match thinking:
        case ThinkingLevel.DEFAULT:
            pass
        case ThinkingLevel.OFF:
            model_settings['thinking'] = False
        case ThinkingLevel.LOW:
            model_settings['thinking'] = 'low'
        case ThinkingLevel.MEDIUM:
            model_settings['thinking'] = 'medium'
        case ThinkingLevel.HIGH:
            model_settings['thinking'] = 'high'
        case _:
            logger.warning(f'Ignoring unknown thinking level "{thinking}".')

    return model_settings
