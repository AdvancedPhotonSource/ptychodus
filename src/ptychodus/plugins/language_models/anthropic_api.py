"""The Anthropic API, reached through the Messages protocol."""

from __future__ import annotations
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ptychodus.api.language_model import (
    LanguageModelEndpoint,
    LanguageModelProvider,
    WireProtocol,
    read_credential,
    require_confidential_url,
)

from ._listing import fetch_json, read_data_ids


class AnthropicOptions(BaseModel):
    base_url: str = Field(
        'https://api.anthropic.com',
        title='Base URL:',
        description=(
            'Anthropic API root URL; a trailing /v1 is tolerated. The API key is read from '
            'ANTHROPIC_API_KEY.'
        ),
    )


class AnthropicLanguageModelProvider(LanguageModelProvider):
    """Reads its key from ``ANTHROPIC_API_KEY``."""

    def get_default_options(self) -> BaseModel:
        return AnthropicOptions()

    def get_default_model(self) -> str:
        return 'claude-sonnet-5-5'

    def list_models(self, options: BaseModel, *, timeout: float = 10.0) -> Sequence[str]:
        base_url, api_key = self._read(options)
        url = f'{base_url}/v1/models'
        headers = {'x-api-key': api_key, 'anthropic-version': '2023-06-01'}
        body = fetch_json(url, headers=headers, params={'limit': '1000'}, timeout=timeout)
        return read_data_ids(url, body)

    def resolve(self, options: BaseModel, model_id: str) -> LanguageModelEndpoint:
        base_url, api_key = self._read(options)
        return LanguageModelEndpoint(
            WireProtocol.ANTHROPIC_MESSAGES, base_url, model_id, api_key=api_key
        )

    def _read(self, options: BaseModel) -> tuple[str, str]:
        if not isinstance(options, AnthropicOptions):
            raise TypeError(f'Expected AnthropicOptions, got {type(options).__name__}.')

        base_url = options.base_url.rstrip('/').removesuffix('/v1')
        require_confidential_url(base_url)
        return base_url, read_credential('ANTHROPIC_API_KEY')
