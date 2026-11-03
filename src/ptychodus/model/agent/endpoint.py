"""The agent's one meeting point with pydantic-ai.

A provider hands back a framework-neutral :class:`LanguageModelEndpoint`; this turns it
into a pydantic-ai model. Replacing the agent framework means replacing this module and
the terminal that drives it, and no provider.
"""

import logging

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ptychodus.api.language_model import LanguageModelEndpoint, WireProtocol, is_confidential_url

from .settings import AgentSettings, ThinkingLevel

logger = logging.getLogger(__name__)

__all__ = [
    'create_chat_model',
    'create_model_settings',
    'get_mcp_server_url',
]


def get_mcp_server_url(settings: AgentSettings) -> str:
    """Return the MCP server URL, or an empty string when unset or insecure."""
    url = settings.mcp_server_url.get_value().strip()

    if url and not is_confidential_url(url):
        logger.warning(f'Ignoring insecure MCP server "{url}"; use an https:// URL.')
        return ''

    return url


def create_chat_model(endpoint: LanguageModelEndpoint) -> Model:
    """Return the pydantic-ai model that speaks *endpoint*'s protocol.

    An endpoint without a key gets a placeholder rather than none: given none, each
    vendor client falls back to its own environment variable, which would send that
    vendor's key to whatever server the endpoint names.
    """
    api_key = endpoint.api_key or 'api-key-not-set'

    match endpoint.protocol:
        case WireProtocol.ANTHROPIC_MESSAGES:
            return AnthropicModel(
                endpoint.model_id,
                provider=AnthropicProvider(base_url=endpoint.base_url, api_key=api_key),
            )
        case WireProtocol.OPENAI_RESPONSES:
            return OpenAIResponsesModel(
                endpoint.model_id,
                provider=OpenAIProvider(base_url=endpoint.base_url, api_key=api_key),
            )
        case WireProtocol.OPENAI_CHAT_COMPLETIONS:
            return OpenAIChatModel(
                endpoint.model_id,
                provider=OpenAIProvider(base_url=endpoint.base_url, api_key=api_key),
            )
        case WireProtocol.GOOGLE_GEMINI:
            return GoogleModel(
                endpoint.model_id,
                provider=GoogleProvider(base_url=endpoint.base_url, api_key=api_key),
            )

    raise ValueError(f'Unsupported wire protocol "{endpoint.protocol}".')


def create_model_settings(settings: AgentSettings) -> ModelSettings:
    """Return generation settings, leaving out whatever the model should choose itself.

    No sampling parameters are sent: current models variously drop, reject or pin
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
