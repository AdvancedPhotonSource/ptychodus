from pathlib import Path
import json

import httpx

from ptychodus.api.common import get_ptychodus_dir

from .account import IRIAccountClient
from .compute import IRIComputeClient
from .facility import IRIFacilityClient
from .status import IRIStatusClient


class IRIClient:
    """See https://api.iri.nersc.gov/#/docs"""

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._api_base_url = api_base_url
        self.account = IRIAccountClient(api_base_url, access_token)
        self.facility = IRIFacilityClient(api_base_url, access_token)
        self.status = IRIStatusClient(api_base_url, access_token)
        self.compute = IRIComputeClient(api_base_url, access_token)

    def get_api_base_url(self) -> str:
        return self._api_base_url

    def print_openapi_specification(self) -> None:
        response = httpx.get(f'{self._api_base_url}/openapi.json', timeout=30.0)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))


def get_iri_tokens_file() -> Path:
    return get_ptychodus_dir() / 'iri_tokens.json'
