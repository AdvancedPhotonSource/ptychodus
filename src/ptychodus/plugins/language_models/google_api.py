"""The Google Gemini API, reached through its native protocol."""

from __future__ import annotations
from collections.abc import Sequence

from pydantic import BaseModel, Field

from ptychodus.api.language_model import (
    LanguageModelEndpoint,
    LanguageModelProvider,
    LanguageModelProviderError,
    WireProtocol,
    read_credential,
    require_confidential_url,
)

from ._listing import fetch_json


class GoogleOptions(BaseModel):
    base_url: str = Field(
        'https://generativelanguage.googleapis.com',
        title='Base URL:',
        description=(
            'Gemini API root URL, without an API version. The API key is read from '
            'GEMINI_API_KEY, falling back to GOOGLE_API_KEY.'
        ),
    )


class GoogleLanguageModelProvider(LanguageModelProvider):
    """Reads its key from ``GEMINI_API_KEY``, falling back to ``GOOGLE_API_KEY``."""

    def get_default_options(self) -> BaseModel:
        return GoogleOptions()

    def get_default_model(self) -> str:
        return ''

    def list_models(self, options: BaseModel, *, timeout: float = 10.0) -> Sequence[str]:
        """Return the models that can chat, without the ``models/`` resource prefix."""
        base_url, api_key = self._read(options)
        url = f'{base_url}/v1beta/models'
        body = fetch_json(
            url, headers={'x-goog-api-key': api_key}, params={'pageSize': '1000'}, timeout=timeout
        )

        try:
            return [
                str(item['name']).removeprefix('models/')
                for item in body['models']
                if 'generateContent' in item.get('supportedGenerationMethods', ())
            ]
        except (AttributeError, KeyError, TypeError) as exc:
            raise LanguageModelProviderError(f'{url} did not answer with a model list.') from exc

    def resolve(self, options: BaseModel, model_id: str) -> LanguageModelEndpoint:
        base_url, api_key = self._read(options)
        return LanguageModelEndpoint(
            WireProtocol.GOOGLE_GEMINI, base_url, model_id, api_key=api_key
        )

    def _read(self, options: BaseModel) -> tuple[str, str]:
        if not isinstance(options, GoogleOptions):
            raise TypeError(f'Expected GoogleOptions, got {type(options).__name__}.')

        base_url = options.base_url.rstrip('/')
        require_confidential_url(base_url)
        return base_url, read_credential('GEMINI_API_KEY', 'GOOGLE_API_KEY')
