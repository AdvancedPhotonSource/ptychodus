#!/usr/bin/env python
"""Login via Globus and store the resulting transfer access token directly."""

import globus_sdk
from globus_sdk.scopes import TransferScopes

from ptychodus.model.genesis.tokens import GenesisAccessTokens, write_tokens
from ptychodus.model.genesis.transfer import get_transfer_tokens_file

CLIENT_ID = '57554043-6dd8-410b-8ece-cd54e9c003bb'
DEFAULT_SCOPE = 'https://auth.globus.org/scopes/08da012c-6998-46f9-9375-a6985ebe3f2b/transfer'
RESOURCE_SERVER = '08da012c-6998-46f9-9375-a6985ebe3f2b'

MAPPED_COLLECTIONS = [
    '9032dd3a-e841-4687-a163-2720da731b5b',
    '05d2c76a-e867-4f67-aa57-76edeb0beda0',
    '3caddd4a-bb35-4c3d-9101-d9a0ad7f3a30',
    '9d6d994a-6d04-11e5-ba46-22000b92c6ec',
]


def main() -> None:
    data_access = [
        globus_sdk.scopes.GCSCollectionScopes(mc).data_access for mc in MAPPED_COLLECTIONS
    ]
    scope = globus_sdk.Scope(DEFAULT_SCOPE).with_dependency(
        TransferScopes.all.with_dependencies(data_access)
    )

    app = globus_sdk.UserApp(
        'amsc-client-login',
        client_id=CLIENT_ID,
        scope_requirements={RESOURCE_SERVER: DEFAULT_SCOPE},
    )
    app.add_scope_requirements({RESOURCE_SERVER: scope})
    app.login()

    access_token = app.token_storage.get_token_data(RESOURCE_SERVER).access_token
    tokens_file = get_transfer_tokens_file()
    write_tokens(tokens_file, [GenesisAccessTokens(facility='AmSC', access_token=access_token)])
    print(f'Token written to {tokens_file}')


if __name__ == '__main__':
    main()
