"""The agent's one meeting point with pydantic-ai's language models.

A provider hands back a framework-neutral :class:`LanguageModelEndpoint`; this turns it
into a pydantic-ai model. Replacing the agent framework means replacing this module and
the agent loop that drives it, and no provider.
"""

from __future__ import annotations

from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from ptychodus.api.language_model import (
    LanguageModelEndpoint,
    LanguageModelProvider,
    LanguageModelProviderError,
    WireProtocol,
)
from ptychodus.api.options import OptionsError, apply_options_document
from ptychodus.api.plugins import Plugin

from .config import Settings, ThinkingLevel

__all__ = [
    'create_chat_model',
    'create_model_settings',
    'resolve_endpoint',
]


def resolve_endpoint(
    settings: Settings, provider: Plugin[LanguageModelProvider]
) -> LanguageModelEndpoint:
    """Resolve the configured language model on `provider`, without touching the network.

    Raises:
        LanguageModelProviderError: if the options are refused, no language model is
            named and the provider has no default, or the provider's credentials are
            missing.
    """
    strategy = provider.strategy

    try:
        options = apply_options_document(
            strategy.get_default_options(),
            settings.provider_options,
            subject=f'PTYCHODUS_AGENT_PROVIDER_OPTIONS for {provider.display_name}',
        )
    except OptionsError as exc:
        raise LanguageModelProviderError(str(exc)) from exc

    model_id = settings.model_id or strategy.get_default_model()

    if not model_id:
        raise LanguageModelProviderError(
            f'{provider.display_name} has no default language model; set PTYCHODUS_AGENT_MODEL_ID.'
        )

    return strategy.resolve(options, model_id)


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


def create_model_settings(thinking: ThinkingLevel, max_tokens: int) -> ModelSettings:
    """Return generation settings, leaving out whatever the language model should choose.

    No sampling parameters are sent: current language models variously drop, reject or
    pin ``temperature`` and ``top_p``. A `max_tokens` of 0 leaves the response length to
    the language model.
    """
    model_settings = ModelSettings()

    if max_tokens > 0:
        model_settings['max_tokens'] = max_tokens

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

    return model_settings
