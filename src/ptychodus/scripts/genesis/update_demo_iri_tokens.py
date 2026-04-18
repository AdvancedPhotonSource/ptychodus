#!/usr/bin/env python
"""Login via Globus and store NERSC and ALCF IRI access tokens directly."""

import globus_sdk
import globus_sdk.gare
from globus_sdk.exc import GlobusAPIError

from ptychodus.model.genesis.iri import get_iri_tokens_file
from ptychodus.model.genesis.tokens import GenesisAccessTokens, write_tokens

NERSC_CLIENT_ID = 'fae5c579-490a-4d76-b6eb-d78f65caeb63'
NERSC_IRI_SCOPE = 'https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api'
NERSC_IRI_RESOURCE_SERVER = 'ed3e577d-f7f3-4639-b96e-ff5a8445d699'

ALCF_AUTH_CLIENT_ID = '8b84fc2d-49e9-49ea-b54d-b3a29a70cf31'
ALCF_SCOPE_CLIENT_ID = '6be511f6-a071-471f-9bc0-02a0d0836723'
ALCF_SCOPE_STRING = f'https://auth.globus.org/scopes/{ALCF_SCOPE_CLIENT_ID}/filesystem'
ALCF_GA_PARAMS = globus_sdk.gare.GlobusAuthorizationParameters(
    session_required_policies=['a128e981-c9a5-417a-97ab-8571c9831bff']
)


def get_nersc_token() -> str:
    client = globus_sdk.NativeAppAuthClient(NERSC_CLIENT_ID)
    client.oauth2_start_flow(requested_scopes=NERSC_IRI_SCOPE)

    print('--- Stage 1: NERSC ---')
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

    return token_response.by_resource_server[NERSC_IRI_RESOURCE_SERVER]['access_token']


def get_alcf_token() -> str:
    print('\n--- Stage 2: ALCF ---')
    app = globus_sdk.UserApp(
        'alcf_facility_api_app',
        client_id=ALCF_AUTH_CLIENT_ID,
        scope_requirements={ALCF_SCOPE_CLIENT_ID: [ALCF_SCOPE_STRING]},
        config=globus_sdk.GlobusAppConfig(request_refresh_tokens=True),
    )
    app.login(auth_params=ALCF_GA_PARAMS)
    auth = app.get_authorizer(ALCF_SCOPE_CLIENT_ID)
    auth.ensure_valid_token()
    return auth.access_token


def main() -> None:
    nersc_token = get_nersc_token()
    alcf_token = get_alcf_token()

    tokens_file = get_iri_tokens_file()
    write_tokens(
        tokens_file,
        [
            GenesisAccessTokens(facility='NERSC', access_token=nersc_token),
            GenesisAccessTokens(facility='ALCF', access_token=alcf_token),
        ],
    )
    print(f'\nTokens written to {tokens_file}')


if __name__ == '__main__':
    main()
