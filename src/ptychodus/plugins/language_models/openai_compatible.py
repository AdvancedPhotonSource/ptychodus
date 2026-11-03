"""Any server speaking OpenAI-compatible Chat Completions: Ollama, vLLM, llama.cpp.

The provider for an instrument without internet access, serving models from a host
on its own network.
"""

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


class OpenAICompatibleOptions(BaseModel):
    base_url: str = Field(
        'http://localhost:11434/v1',
        title='Base URL:',
        description='Server base URL, including its /v1.',
    )
    api_key_env_var: str = Field(
        '',
        title='API Key Variable:',
        description=(
            "Environment variable holding the server's API key; leave empty for a server "
            'that takes none. With a key, the URL must be https:// or loopback http://.'
        ),
    )


class OpenAICompatibleLanguageModelProvider(LanguageModelProvider):
    """Sends a key only when ``api_key_env_var`` names one.

    Without a key a cleartext URL on the local network is accepted, since nothing is
    disclosed by sending it a request.
    """

    def get_default_options(self) -> BaseModel:
        return OpenAICompatibleOptions()

    def get_default_model(self) -> str:
        return ''

    def list_models(self, options: BaseModel, *, timeout: float = 10.0) -> Sequence[str]:
        base_url, api_key = self._read(options)
        url = f'{base_url}/models'
        headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        return read_data_ids(url, fetch_json(url, headers=headers, timeout=timeout))

    def resolve(self, options: BaseModel, model_id: str) -> LanguageModelEndpoint:
        base_url, api_key = self._read(options)
        return LanguageModelEndpoint(
            WireProtocol.OPENAI_CHAT_COMPLETIONS, base_url, model_id, api_key=api_key
        )

    def _read(self, options: BaseModel) -> tuple[str, str]:
        if not isinstance(options, OpenAICompatibleOptions):
            raise TypeError(f'Expected OpenAICompatibleOptions, got {type(options).__name__}.')

        base_url = options.base_url.rstrip('/')
        env_var = options.api_key_env_var.strip()

        if not env_var:
            return base_url, ''

        require_confidential_url(base_url)
        return base_url, read_credential(env_var)
