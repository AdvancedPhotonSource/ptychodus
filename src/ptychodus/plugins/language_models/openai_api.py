"""The OpenAI API, reached through the Responses protocol."""

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


class OpenAIOptions(BaseModel):
    base_url: str = Field(
        'https://api.openai.com/v1',
        title='Base URL:',
        description='OpenAI API base URL, including its /v1. The API key is read from OPENAI_API_KEY.',
    )


class OpenAILanguageModelProvider(LanguageModelProvider):
    """Reads its key from ``OPENAI_API_KEY``."""

    def get_default_options(self) -> BaseModel:
        return OpenAIOptions()

    def get_default_model(self) -> str:
        return ''

    def list_models(self, options: BaseModel, *, timeout: float = 10.0) -> Sequence[str]:
        base_url, api_key = self._read(options)
        url = f'{base_url}/models'
        headers = {'Authorization': f'Bearer {api_key}'}
        return read_data_ids(url, fetch_json(url, headers=headers, timeout=timeout))

    def resolve(self, options: BaseModel, model_id: str) -> LanguageModelEndpoint:
        base_url, api_key = self._read(options)
        return LanguageModelEndpoint(
            WireProtocol.OPENAI_RESPONSES, base_url, model_id, api_key=api_key
        )

    def _read(self, options: BaseModel) -> tuple[str, str]:
        if not isinstance(options, OpenAIOptions):
            raise TypeError(f'Expected OpenAIOptions, got {type(options).__name__}.')

        base_url = options.base_url.rstrip('/')
        require_confidential_url(base_url)
        return base_url, read_credential('OPENAI_API_KEY')
