from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel
import requests

from .token_storage import (
    GenesisAccessTokens,
    create_headers,
    get_transfer_access_tokens_file,
    write_access_tokens,
)


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
        self._base_url = f'{api_base_url}/transfer'
        self._headers = create_headers(access_token)

    def check_auth_token(self) -> str:
        response = requests.get(f'{self._base_url}/auth/globus', headers=self._headers)
        response.raise_for_status()
        return response.text

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


if __name__ == '__main__':
    tokens_file = get_transfer_access_tokens_file()
    access_tokens: list[GenesisAccessTokens] = []

    while True:
        name = input('Enter a name for the access token (or blank to finish): ').strip()

        if not name:
            break

        api_base_url = input('Enter the API base URL: ').strip()
        access_token = input('Enter the access token: ').strip()
        access_tokens.append(
            GenesisAccessTokens(
                name=name,
                api_base_url=api_base_url,
                access_token=access_token,
            )
        )

    write_access_tokens(tokens_file, access_tokens)
