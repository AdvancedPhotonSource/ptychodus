from pathlib import Path
from typing import Any
import json
import logging

import requests

from ptychodus.api.common import get_ptychodus_dir

from ..tokens import GenesisAccessTokens, read_tokens, write_tokens
from .compute import IRIComputeClient
from .facility import IRIFacilityClient
from .status import IRIStatusClient

logger = logging.getLogger(__name__)


class IRIClient:
    def __init__(self, api_base_url: str, access_token: str) -> None:
        self.facility = IRIFacilityClient(api_base_url, access_token)
        self.status = IRIStatusClient(api_base_url, access_token)
        self.compute = IRIComputeClient(api_base_url, access_token)


def get_iri_tokens_file() -> Path:
    return get_ptychodus_dir() / 'iri_tokens.json'


def set_iri_tokens_cli() -> None:
    tokens_file = get_iri_tokens_file()
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

    write_tokens(tokens_file, access_tokens)


def check_iri_tokens_cli() -> None:
    logging.basicConfig(level=logging.INFO)

    tokens_file = get_iri_tokens_file()
    access_tokens = read_tokens(tokens_file)

    for token in access_tokens:
        client = IRIClient(token.api_base_url, token.access_token)
        data: dict[str, Any] = {}

        try:
            facility = client.facility.get_facility()
            sites = client.facility.get_sites()
            resources = client.status.get_resources()
        except requests.HTTPError as exc:
            logger.error(f'"{token.name}" token error: {exc}')
        else:
            data[token.name] = {
                'facility': facility.model_dump(mode='json'),
                'sites': [site.model_dump(mode='json') for site in sites],
                'resources': [resource.model_dump(mode='json') for resource in resources],
            }

        print(json.dumps(data, indent=4))
