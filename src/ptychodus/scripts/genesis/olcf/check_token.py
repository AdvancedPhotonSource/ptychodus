#!/usr/bin/env python

import json
import logging
import sys

import requests

from ptychodus.model.genesis.iri.client import get_iri_tokens_file
from ptychodus.model.genesis.tokens import create_headers, load_tokens

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    tokens_file = get_iri_tokens_file()
    access_tokens = load_tokens(tokens_file)

    olcf_token = next(
        (t.access_token for t in access_tokens if t.facility.casefold() == 'olcf'), None
    )

    if olcf_token is None:
        logger.error('No OLCF token found in %s', tokens_file)
        sys.exit(1)

    headers = create_headers(olcf_token)

    logger.info('Testing token validity...')
    token_response = requests.get(
        'https://s3m.olcf.ornl.gov/olcf/v1/token/ctls/introspect', headers=headers
    )
    print(json.dumps(token_response.json(), indent=2))

    logger.info('Testing resource availability...')
    resource_response = requests.get(
        'https://amsc-open.s3m.olcf.ornl.gov/api/v1/status/resources/odo', headers=headers
    )
    print(json.dumps(resource_response.json(), indent=2))


if __name__ == '__main__':
    main()
