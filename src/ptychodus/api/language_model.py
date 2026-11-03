"""Language-model providers, and the endpoint each hands an agent.

A provider is a service that hosts chat models: a commercial API, a facility gateway,
or a server on the instrument network. It knows where the service is, which wire
protocol each of its models speaks, which models it offers and where its credential
comes from. What it hands back is a :class:`LanguageModelEndpoint`, plain data that any
agent framework or a hand-written HTTP loop can consume, so nothing here assumes one.

Instruments often run on networks without internet access, so a provider touches the
network only in :meth:`LanguageModelProvider.list_models`; constructing one, describing
its options and resolving an endpoint are all local.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from urllib.parse import urlparse
import os

from pydantic import BaseModel

__all__ = [
    'LanguageModelEndpoint',
    'LanguageModelProvider',
    'LanguageModelProviderError',
    'WireProtocol',
    'is_confidential_url',
    'read_credential',
    'require_confidential_url',
]


class WireProtocol(StrEnum):
    """The request and response format a model is reached through."""

    ANTHROPIC_MESSAGES = 'anthropic-messages'
    OPENAI_RESPONSES = 'openai-responses'
    OPENAI_CHAT_COMPLETIONS = 'openai-chat-completions'
    GOOGLE_GEMINI = 'google-gemini'


@dataclass(frozen=True)
class LanguageModelEndpoint:
    """Everything needed to send one model a request.

    *base_url* is the root the protocol's client expects: the one the vendor's own SDK
    takes as its base URL. *api_key* is empty when the service takes none, and is left
    out of the ``repr`` so an endpoint can be logged without disclosing it.
    """

    protocol: WireProtocol
    base_url: str
    model_id: str
    api_key: str = field(default='', repr=False)


class LanguageModelProviderError(Exception):
    """A provider cannot be used as configured: a credential is missing, a URL is
    refused, or the service answered with something other than a model list."""


class LanguageModelProvider(ABC):
    """A service hosting chat models, and the options it is reached with.

    Stateless and configured per call, like the generators: options are plain pydantic
    edited by the caller and passed back. A provider does not name itself; the name a
    person selects it by is the one it was registered under.

    A credential is read from the environment when it is needed and is never one of the
    options, so a settings file can travel with a dataset without carrying it.
    """

    @abstractmethod
    def get_default_options(self) -> BaseModel:
        """Return this provider's options, populated with their defaults."""

    @abstractmethod
    def get_default_model(self) -> str:
        """Return the model to start from, or an empty string when there is none."""

    @abstractmethod
    def list_models(self, options: BaseModel, *, timeout: float = 10.0) -> Sequence[str]:
        """Return the IDs of the models the service offers.

        The only method that touches the network; *timeout* is in seconds.

        Raises:
            LanguageModelProviderError: if the provider cannot be used as configured or
                the service's answer cannot be read.
            httpx.HTTPError: if the request fails.
        """

    @abstractmethod
    def resolve(self, options: BaseModel, model_id: str) -> LanguageModelEndpoint:
        """Return the endpoint that reaches *model_id*, without touching the network.

        Raises:
            LanguageModelProviderError: if the provider cannot be used as configured.
        """


def is_confidential_url(url: str) -> bool:
    """Return whether *url* is ``https`` or ``http`` on the loopback interface."""
    parsed = urlparse(url)

    if parsed.scheme == 'https':
        return True

    return parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1', '::1'}


def require_confidential_url(url: str) -> None:
    """Refuse to send a credential to *url* unless it is confidential.

    A base URL is a plain string in a settings file that may have arrived with a
    dataset, so a credential goes only where it cannot be read in transit.

    Raises:
        LanguageModelProviderError: if *url* is neither ``https`` nor loopback ``http``.
    """
    if not is_confidential_url(url):
        raise LanguageModelProviderError(
            f'Refusing to send a credential to insecure endpoint "{url}"; use an https:// URL.'
        )


def read_credential(*env_vars: str) -> str:
    """Return the value of the first of *env_vars* that is set and not empty.

    Raises:
        LanguageModelProviderError: naming every variable tried, if none is set.
    """
    for env_var in env_vars:
        value = os.environ.get(env_var, '')

        if value:
            return value

    raise LanguageModelProviderError(f'Set {" or ".join(env_vars)} to use this provider.')
