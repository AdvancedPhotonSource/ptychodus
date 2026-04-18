#!/usr/bin/env python

from typing import Any
import argparse
import json
import logging

import requests

from ptychodus.model.genesis.core import create_facility_adapters
from ptychodus.model.genesis.iri import get_iri_tokens_file
from ptychodus.model.genesis.tokens import GenesisAccessTokens, write_tokens

logger = logging.getLogger(__name__)


def set_tokens() -> None:
    tokens_file = get_iri_tokens_file()
    access_tokens: list[GenesisAccessTokens] = []

    while True:
        facility = input(
            'Enter a facility name for the access token (or blank to finish): '
        ).strip()

        if not facility:
            break

        access_token = input('Enter the access token: ').strip()
        access_tokens.append(
            GenesisAccessTokens(
                facility=facility,
                access_token=access_token,
            )
        )

    write_tokens(tokens_file, access_tokens)


def check_tokens() -> None:
    logging.basicConfig(level=logging.INFO)

    adapters = create_facility_adapters()
    data: dict[str, Any] = {}

    for name, adapter in adapters.items():
        logger.info(f'Checking IRI access token for facility "{name}"...')
        client = adapter.get_iri_client()

        try:
            facility = client.facility.get_facility()
            sites = client.facility.get_sites()
            resources = client.status.get_resources()
        except requests.HTTPError as exc:
            logger.error(f'"{name}" token error: {exc}')
        else:
            data[name] = {
                'facility': facility.model_dump(mode='json'),
                'sites': [site.model_dump(mode='json') for site in sites],
                'resources': [resource.model_dump(mode='json') for resource in resources],
            }

    print(json.dumps(data, indent=4))


def main() -> None:
    parser = argparse.ArgumentParser(description='Store and check transfer access tokens.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    set_parser = subparsers.add_parser('set', help='Set transfer access tokens')
    set_parser.set_defaults(func=set_tokens)

    check_parser = subparsers.add_parser('check', help='Check transfer access tokens')
    check_parser.set_defaults(func=check_tokens)

    args = parser.parse_args()
    args.func()


if __name__ == '__main__':
    main()
