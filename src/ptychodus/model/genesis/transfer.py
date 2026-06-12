from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final
import json

from pydantic import BaseModel
import requests

from ptychodus.api.common import get_ptychodus_dir

from .tokens import create_headers


class TransferStatus(StrEnum):
    ACTIVE = 'ACTIVE'
    SUCCEEDED = 'SUCCEEDED'
    FAILED = 'FAILED'
    CANCELED = 'CANCELED'
    UNKNOWN = 'UNKNOWN'


class GlobusTransferInputs(BaseModel):
    source_url: str | None = None
    destination_url: str | None = None
    label: str = ''
    source_uuid: str | None = None
    source_path: str | None = None
    destination_uuid: str | None = None
    destination_path: str | None = None


class GlobusTransferResult(BaseModel):
    label: str | None = None
    transfer_uuid: str
    status: TransferStatus | None = None
    completion_time: datetime | None = None
    bytes_transferred: int | None = None
    effective_bytes_per_second: int | None = None


class AmSCGlobusTransferClient:
    """See https://amsc-data-api.nersc.gov/docs#/Globus"""

    NAME: Final[str] = 'AmSC'

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._api_base_url = api_base_url.rstrip('/')
        self._headers = create_headers(access_token)

    def check_auth_token(self) -> Mapping[str, Any]:
        response = requests.get(f'{self._api_base_url}/movement/auth/globus', headers=self._headers)
        response.raise_for_status()
        return response.json()

    def start_transfer(self, inputs: GlobusTransferInputs) -> GlobusTransferResult:
        response = requests.post(
            f'{self._api_base_url}/movement/transfer/globus',
            json=inputs.model_dump(mode='json'),
            headers=self._headers,
        )
        response.raise_for_status()
        return GlobusTransferResult.model_validate(response.json())

    def get_transfer(self, transfer_id: str) -> GlobusTransferResult:
        response = requests.get(
            f'{self._api_base_url}/movement/transfer/globus/{transfer_id}',
            headers=self._headers,
        )
        response.raise_for_status()
        return GlobusTransferResult.model_validate(response.json())

    def delete_transfer(self, transfer_id: str) -> GlobusTransferResult:
        response = requests.delete(
            f'{self._api_base_url}/movement/transfer/globus/{transfer_id}',
            headers=self._headers,
        )
        response.raise_for_status()
        return GlobusTransferResult.model_validate(response.json())

    def print_openapi_specification(self) -> None:
        response = requests.get(f'{self._api_base_url}/openapi.json')
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


def get_amsc_transfer_api_url() -> str:
    return 'https://amsc-data-api.nersc.gov'


def get_transfer_tokens_file() -> Path:
    return get_ptychodus_dir() / 'genesis_transfer_tokens.json'
