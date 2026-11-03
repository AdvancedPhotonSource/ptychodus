"""Language-model providers for the agent, one per service hosting chat models.

None imports an LLM SDK or touches the network to register, so every provider is
offered whether or not its service can be reached from this machine.
"""

from ptychodus.api.plugins import PluginRegistry

from .anthropic_api import AnthropicLanguageModelProvider
from .argo import ArgoLanguageModelProvider
from .google_api import GoogleLanguageModelProvider
from .openai_api import OpenAILanguageModelProvider
from .openai_compatible import OpenAICompatibleLanguageModelProvider

__all__ = [
    'AnthropicLanguageModelProvider',
    'ArgoLanguageModelProvider',
    'GoogleLanguageModelProvider',
    'OpenAICompatibleLanguageModelProvider',
    'OpenAILanguageModelProvider',
]


def register_plugins(registry: PluginRegistry) -> None:
    registry.language_model_providers.register_plugin(
        AnthropicLanguageModelProvider(), simple_name='Anthropic', display_name='Anthropic'
    )
    registry.language_model_providers.register_plugin(
        ArgoLanguageModelProvider(), simple_name='Argo', display_name='Argo (ANL)'
    )
    registry.language_model_providers.register_plugin(
        GoogleLanguageModelProvider(), simple_name='Google', display_name='Google Gemini'
    )
    registry.language_model_providers.register_plugin(
        OpenAILanguageModelProvider(), simple_name='OpenAI', display_name='OpenAI'
    )
    registry.language_model_providers.register_plugin(
        OpenAICompatibleLanguageModelProvider(),
        simple_name='OpenAICompatible',
        display_name='OpenAI-compatible (local)',
    )
