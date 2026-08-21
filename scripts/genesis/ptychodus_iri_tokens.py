#!/usr/bin/env python

from getpass import getpass
from typing import Any
import argparse
import json
import logging

import httpx

from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.genesis.core import create_facility_adapters
from ptychodus.model.genesis.iri import get_iri_tokens_file
from ptychodus.model.genesis.settings import GenesisSettings
from ptychodus.model.genesis.tokens import GenesisAccessTokens, save_tokens

logger = logging.getLogger(__name__)


def store_tokens() -> None:
    tokens_file = get_iri_tokens_file()
    access_tokens: list[GenesisAccessTokens] = []

    while True:
        facility = input(
            'Enter a facility name for the access token (or blank to finish): '
        ).strip()

        if not facility:
            break

        access_token = getpass('Enter the access token: ').strip()
        access_tokens.append(
            GenesisAccessTokens(
                facility=facility,
                access_token=access_token,
            )
        )

    save_tokens(tokens_file, access_tokens)


def check_tokens() -> None:
    logging.basicConfig(level=logging.INFO)

    settings_registry = SettingsRegistry()
    settings = GenesisSettings(settings_registry)
    adapters = create_facility_adapters(settings)
    data = list()

    for name, adapter in adapters.items():
        logger.info(f'Checking IRI access token for facility "{name}"...')
        client = adapter.get_iri_client()

        try:
            projects = client.account.get_projects()
        except httpx.HTTPStatusError as exc:
            logger.error(f'"{name}" token error: {exc}')
        else:
            data = [project.model_dump(mode='json') for project in projects]

    print(json.dumps(data, indent=4))


def list_resources() -> None:
    logging.basicConfig(level=logging.INFO)

    settings_registry = SettingsRegistry()
    settings = GenesisSettings(settings_registry)
    adapters = create_facility_adapters(settings)
    data: dict[str, Any] = {}

    for name, adapter in adapters.items():
        logger.info(f'Checking IRI access token for facility "{name}"...')
        client = adapter.get_iri_client()

        try:
            facility = client.facility.get_facility()
            sites = client.facility.get_sites()
            resources = client.status.get_resources()
        except httpx.HTTPStatusError as exc:
            logger.error(f'"{name}" token error: {exc}')
        else:
            data[name] = {
                'facility': facility.model_dump(mode='json'),
                'sites': [site.model_dump(mode='json') for site in sites],
                'resources': [resource.model_dump(mode='json') for resource in resources],
            }

    print(json.dumps(data, indent=4))


def print_spec(facility: str) -> None:
    settings_registry = SettingsRegistry()
    settings = GenesisSettings(settings_registry)
    adapters = create_facility_adapters(settings)

    try:
        adapter = adapters[facility]
    except KeyError:
        known = ', '.join(sorted(adapters))
        raise SystemExit(f'Unknown facility "{facility}". Known facilities: {known}')

    client = adapter.get_iri_client()
    client.print_openapi_specification()


def main() -> None:
    parser = argparse.ArgumentParser(description='Store and check transfer access tokens.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    set_parser = subparsers.add_parser('store', help='Store transfer access tokens')
    set_parser.set_defaults(func=lambda _: store_tokens())

    check_parser = subparsers.add_parser('check', help='Check transfer access tokens')
    check_parser.set_defaults(func=lambda _: check_tokens())

    resources_parser = subparsers.add_parser('resources', help='List facility resources')
    resources_parser.set_defaults(func=lambda _: list_resources())

    spec_parser = subparsers.add_parser('spec', help='Print OpenAPI spec for a facility')
    spec_parser.add_argument('facility', help='Facility name')
    spec_parser.set_defaults(func=lambda args: print_spec(args.facility))

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
