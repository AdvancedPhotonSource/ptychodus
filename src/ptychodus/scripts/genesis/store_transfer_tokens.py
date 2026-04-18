#!/usr/bin/env python

import argparse
import json
import logging

import requests

from ptychodus.model.genesis.core import create_globus_transfer_providers
from ptychodus.model.genesis.tokens import GenesisAccessTokens, write_tokens
from ptychodus.model.genesis.transfer import get_transfer_tokens_file

logger = logging.getLogger(__name__)


def set_tokens() -> None:
    tokens_file = get_transfer_tokens_file()
    access_token = GenesisAccessTokens(
        facility='AmSC',
        access_token=input('Enter the access token: ').strip(),
    )
    write_tokens(tokens_file, [access_token])


def check_tokens() -> None:
    logging.basicConfig(level=logging.INFO)

    providers = create_globus_transfer_providers()

    for name, client in providers.items():
        logger.info(f'Checking transfer access token for provider "{name}"...')

        try:
            data = client.check_auth_token()
        except requests.HTTPError as exc:
            logger.error(f'"{name}" token error: {exc}')
        else:
            logger.info(f'"{name}" token response:' + json.dumps(data, indent=4))


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
