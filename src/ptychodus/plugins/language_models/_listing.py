"""Reading a model list out of a service's JSON answer."""

from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from ptychodus.api.language_model import LanguageModelProviderError


def fetch_json(
    url: str, *, headers: Mapping[str, str], params: Mapping[str, str] | None = None, timeout: float
) -> Any:
    """Return the decoded JSON body of a GET on *url*.

    Raises:
        httpx.HTTPError: if the request fails or answers with an error status.
        LanguageModelProviderError: if the body is not JSON.
    """
    response = httpx.get(url, headers=dict(headers), params=params, timeout=timeout)
    response.raise_for_status()

    try:
        return response.json()
    except ValueError as exc:
        raise LanguageModelProviderError(f'{url} did not answer with JSON.') from exc


def read_data_ids(url: str, body: Any) -> Sequence[str]:
    """Return ``body['data'][i]['id']``, the shape OpenAI and Anthropic both list in."""
    try:
        return [str(item['id']) for item in body['data']]
    except (KeyError, TypeError) as exc:
        raise LanguageModelProviderError(f'{url} did not answer with a model list.') from exc
