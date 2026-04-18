#!/usr/bin/env python
"""Login via Globus and store the NERSC IRI access token directly."""

import globus_sdk
from globus_sdk.exc import GlobusAPIError

from ptychodus.model.genesis.iri import get_iri_tokens_file
from ptychodus.model.genesis.tokens import GenesisAccessTokens, write_tokens

CLIENT_ID = 'fae5c579-490a-4d76-b6eb-d78f65caeb63'
NERSC_IRI_SCOPE = 'https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api'
NERSC_IRI_RESOURCE_SERVER = 'ed3e577d-f7f3-4639-b96e-ff5a8445d699'


def main() -> None:
    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    client.oauth2_start_flow(requested_scopes=NERSC_IRI_SCOPE)

    print('Open this URL, login, and consent:')
    print(client.oauth2_get_authorize_url())
    code = input('\nEnter authorization code: ').strip()
    if not code:
        raise RuntimeError(
            'No authorization code entered. Re-run the script and paste the code '
            'shown by Globus after login.'
        )

    try:
        token_response = client.oauth2_exchange_code_for_tokens(code)
    except GlobusAPIError as exc:
        if exc.http_status == 400:
            raise RuntimeError(
                'Authorization code exchange failed. The code was empty, invalid, '
                'expired, or already used. Re-run the script and complete the '
                'Globus login flow again.'
            ) from exc
        raise RuntimeError(
            f'Authorization code exchange failed with HTTP {exc.http_status}. '
            'Re-run the script and try again.'
        ) from exc

    access_token = token_response.by_resource_server[NERSC_IRI_RESOURCE_SERVER]['access_token']
    tokens_file = get_iri_tokens_file()
    write_tokens(tokens_file, [GenesisAccessTokens(facility='NERSC', access_token=access_token)])
    print(f'Token written to {tokens_file}')


if __name__ == '__main__':
    main()
