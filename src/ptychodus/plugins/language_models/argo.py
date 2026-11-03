"""Argo, the Argonne gateway to commercial chat models.

Argo serves three native APIs under one gateway root: Anthropic Messages
(``/v1/messages``), OpenAI Responses (``/v1/responses``) and OpenAI-compatible Chat
Completions (``/v1/chat/completions``). Each takes the caller's ANL domain username as
its bearer token in place of an API key. Reachable only from the ANL network.
"""

from __future__ import annotations
from collections.abc import Sequence
import getpass
import os

from pydantic import BaseModel, Field

from ptychodus.api.language_model import (
    LanguageModelEndpoint,
    LanguageModelProvider,
    LanguageModelProviderError,
    WireProtocol,
    require_confidential_url,
)

from ._listing import fetch_json, read_data_ids


class ArgoOptions(BaseModel):
    base_url: str = Field(
        'https://apps.inside.anl.gov/argoapi',
        title='Base URL:',
        description=(
            'Argo Gateway root URL; a trailing /v1 is tolerated. Your ANL domain username '
            'is read from ARGO_USER, falling back to the login name.'
        ),
    )


def get_gateway_url(base_url: str) -> str:
    """Return the gateway root, tolerating a trailing ``/`` or ``/v1``."""
    return base_url.rstrip('/').removesuffix('/v1')


def get_argo_user(url: str) -> str:
    """Return the ANL domain username sent to *url* as its bearer token.

    Reads ``ARGO_USER``, falling back to the login name. The value is a username (or an
    ``svc``-prefixed service account), never an email address. It is not a secret, but
    it attributes usage and cost, so it is sent only to a confidential URL.

    Raises:
        LanguageModelProviderError: if *url* is neither ``https`` nor loopback ``http``,
            or no username can be determined.
    """
    require_confidential_url(url)
    user = os.environ.get('ARGO_USER', '')

    if not user:
        try:
            user = getpass.getuser()
        except OSError as exc:
            raise LanguageModelProviderError('Set ARGO_USER to your ANL domain username.') from exc

    return user


class ArgoLanguageModelProvider(LanguageModelProvider):
    def get_default_options(self) -> BaseModel:
        return ArgoOptions()

    def get_default_model(self) -> str:
        return 'claudesonnet55'

    def list_models(self, options: BaseModel, *, timeout: float = 10.0) -> Sequence[str]:
        url = f'{self._get_gateway_url(options)}/v1/models'
        headers = {'Authorization': f'Bearer {get_argo_user(url)}'}
        return read_data_ids(url, fetch_json(url, headers=headers, timeout=timeout))

    def resolve(self, options: BaseModel, model_id: str) -> LanguageModelEndpoint:
        """Route *model_id* to its vendor-native API.

        ``claude*`` models use Anthropic Messages, ``gpt*`` models OpenAI Responses, and
        every other model OpenAI-compatible Chat Completions.
        """
        gateway_url = self._get_gateway_url(options)
        user = get_argo_user(gateway_url)

        if model_id.startswith('claude'):
            return LanguageModelEndpoint(
                WireProtocol.ANTHROPIC_MESSAGES, gateway_url, model_id, api_key=user
            )

        protocol = (
            WireProtocol.OPENAI_RESPONSES
            if model_id.startswith('gpt')
            else WireProtocol.OPENAI_CHAT_COMPLETIONS
        )
        return LanguageModelEndpoint(protocol, f'{gateway_url}/v1', model_id, api_key=user)

    def _get_gateway_url(self, options: BaseModel) -> str:
        if not isinstance(options, ArgoOptions):
            raise TypeError(f'Expected ArgoOptions, got {type(options).__name__}.')

        return get_gateway_url(options.base_url)
