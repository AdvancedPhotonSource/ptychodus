from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
import json
import logging

from pydantic import BaseModel
import requests

from .tokens import (
    GenesisAccessTokens,
    create_headers,
    get_transfer_tokens_file,
    read_tokens,
    write_tokens,
)

logger = logging.getLogger(__name__)


class GlobusTransferInputs(BaseModel):
    source_url: str | None = None
    destination_url: str | None = None
    label: str = ''
    source_uuid: str | None = None
    source_path: str | None = None
    destination_uuid: str | None = None
    destination_path: str | None = None


class GlobusTransferDetails(BaseModel):
    name: str
    source_url: str  # uri >= 1 characters
    destination_url: str  # uri >= 1 characters
    transfer_uuid: str | None


class GlobusTransferResult(BaseModel):
    transfer_uuid: str
    status: str | None = None
    completion_time: datetime | None = None
    destination_url: str | None
    reason: str | None
    bytes_transferred: int | None = None
    effective_bytes_per_second: int | None = None


class GenesisGlobusTransferClient:
    # See https://amsc-data-api.nersc.gov/docs#/Globus

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._base_url = api_base_url.rstrip('/') + '/transfer'
        self._headers = create_headers(access_token)

    def check_auth_token(self) -> Mapping[str, Any]:
        response = requests.get(f'{self._base_url}/auth/globus', headers=self._headers)
        response.raise_for_status()
        return response.json()

    def start_transfer(self, inputs: GlobusTransferInputs) -> Sequence[GlobusTransferDetails]:
        response = requests.post(
            f'{self._base_url}/globus',
            json=inputs.model_dump(mode='json'),
            headers=self._headers,
        )
        response.raise_for_status()
        return [GlobusTransferDetails.model_validate(item) for item in response.json()]

    def get_transfer(self, transfer_id: str) -> GlobusTransferResult:
        response = requests.get(
            f'{self._base_url}/globus/{transfer_id}',
            headers=self._headers,
        )
        response.raise_for_status()
        return GlobusTransferResult.model_validate(response.json())

    def delete_transfer(self, transfer_id: str) -> GlobusTransferResult:
        response = requests.delete(
            f'{self._base_url}/globus/{transfer_id}',
            headers=self._headers,
        )
        response.raise_for_status()
        return GlobusTransferResult.model_validate(response.json())


def set_transfer_tokens_cli() -> None:
    logging.basicConfig(level=logging.INFO)

    tokens_file = get_transfer_tokens_file()
    access_tokens: list[GenesisAccessTokens] = []

    while True:
        name = input('Enter a name for the access token (or blank to finish): ').strip()

        if not name:
            break

        api_base_url = input('Enter the API base URL: ').strip()
        access_token = input('Enter the access token: ').strip()

        client = GenesisGlobusTransferClient(api_base_url, access_token)
        data = client.check_auth_token()
        logger.info(f'Check auth token response: {data}')

        access_tokens.append(
            GenesisAccessTokens(
                name=name,
                api_base_url=api_base_url,
                access_token=access_token,
            )
        )

    write_tokens(tokens_file, access_tokens)


def check_transfer_tokens_cli() -> None:
    logging.basicConfig(level=logging.INFO)

    tokens_file = get_transfer_tokens_file()
    access_tokens = read_tokens(tokens_file)

    for token in access_tokens:
        client = GenesisGlobusTransferClient(token.api_base_url, token.access_token)

        try:
            data = client.check_auth_token()
        except requests.HTTPError as exc:
            logger.error(f'Token "{token.name}" error: {exc}')
        else:
            logger.info(f'Token "{token.name}" response:' + json.dumps(data, indent=4))
