#!/usr/bin/env python

from getpass import getpass
import argparse
import json
import logging

import httpx

from ptychodus.model.genesis.core import create_globus_transfer_providers
from ptychodus.model.genesis.tokens import GenesisAccessTokens, save_tokens
from ptychodus.model.genesis.transfer import get_transfer_tokens_file

logger = logging.getLogger(__name__)


def store_tokens() -> None:
    tokens_file = get_transfer_tokens_file()
    access_token = GenesisAccessTokens(
        facility='AmSC',
        access_token=getpass('Enter the access token: ').strip(),
    )
    save_tokens(tokens_file, [access_token])


def check_tokens() -> None:
    logging.basicConfig(level=logging.INFO)

    providers = create_globus_transfer_providers()

    for name, client in providers.items():
        logger.info(f'Checking transfer access token for provider "{name}"...')

        try:
            data = client.check_auth_token()
        except httpx.HTTPStatusError as exc:
            logger.error(f'"{name}" token error: {exc}')
        else:
            logger.info(f'"{name}" token response:' + json.dumps(data, indent=4))


def print_spec(provider: str) -> None:
    providers = create_globus_transfer_providers()

    try:
        client = providers[provider]
    except KeyError:
        known = ', '.join(sorted(providers))
        raise SystemExit(f'Unknown provider "{provider}". Known providers: {known}')

    client.print_openapi_specification()


def main() -> None:
    parser = argparse.ArgumentParser(description='Store and check transfer access tokens.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    set_parser = subparsers.add_parser('store', help='Store transfer access tokens')
    set_parser.set_defaults(func=lambda _: store_tokens())

    check_parser = subparsers.add_parser('check', help='Check transfer access tokens')
    check_parser.set_defaults(func=lambda _: check_tokens())

    spec_parser = subparsers.add_parser('spec', help='Print OpenAPI spec for a provider')
    spec_parser.add_argument('provider', help='Provider name')
    spec_parser.set_defaults(func=lambda args: print_spec(args.provider))

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
