#!/usr/bin/env python3
"""
Globus token helpers for ALCF IRI, NERSC IRI, and NERSC Transfer.

    from claude_token import get_alcf_iri_token
    from claude_token import get_alcf_transfer_token
    from claude_token import get_nersc_iri_token
    from claude_token import get_nersc_transfer_token
    from claude_token import get_olcf_transfer_token
"""

import json
import os
import stat
import time
from pathlib import Path

import globus_sdk
from globus_sdk import Scope
from globus_sdk.login_flows import LocalServerLoginFlowManager  # noqa: F401  # side-effect: loads globus_sdk.gare
from globus_sdk.exc import GlobusAPIError
from globus_sdk.scopes import TransferScopes, GCSCollectionScopes

ALCF_APP_NAME = 'alcf_facility_api_app'
ALCF_AUTH_CLIENT_ID = '8b84fc2d-49e9-49ea-b54d-b3a29a70cf31'
ALCF_SCOPE_CLIENT_ID = '6be511f6-a071-471f-9bc0-02a0d0836723'
ALCF_SCOPE_STRING = f'https://auth.globus.org/scopes/{ALCF_SCOPE_CLIENT_ID}/filesystem'
ALCF_TOKENS_PATH = (
    Path.home() / '.globus' / 'app' / ALCF_AUTH_CLIENT_ID / ALCF_APP_NAME / 'tokens.json'
)
ALCF_GA_PARAMS = globus_sdk.gare.GlobusAuthorizationParameters(
    session_required_policies=['a128e981-c9a5-417a-97ab-8571c9831bff']
)

NATIVE_APP_CLIENT_ID = '7e71ebc9-3967-47e3-a1cf-c5f164b8a816'
NERSC_RESOURCE_SERVER = 'auth.globus.org'
NERSC_IRI_SCOPE = 'https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api'
NERSC_REQUIRED_SCOPES = {
    'openid',
    'profile',
    'email',
    'urn:globus:auth:scope:auth.globus.org:view_identities',
}
NERSC_REQUESTED_SCOPES = NERSC_REQUIRED_SCOPES | {NERSC_IRI_SCOPE}
NERSC_DEFAULT_TOKEN_FILE = Path.home() / '.globus' / 'auth_tokens.json'

NERSC_TRANSFER_COLLECTIONS = [
    '5049be87-dfcb-4ea3-84d9-07b11647f5d2',
    '9d6d994a-6d04-11e5-ba46-22000b92c6ec',
]
NERSC_AMSC_RESOURCE_SERVER = '08da012c-6998-46f9-9375-a6985ebe3f2b'
NERSC_TRANSFER_SCOPE_URL = f'https://auth.globus.org/scopes/{NERSC_AMSC_RESOURCE_SERVER}/transfer'
NERSC_TRANSFER_DEFAULT_TOKEN_FILE = Path.home() / '.globus' / 'nersc_transfer_tokens.json'

ALCF_TRANSFER_COLLECTIONS = [
    '5049be87-dfcb-4ea3-84d9-07b11647f5d2',
    '05d2c76a-e867-4f67-aa57-76edeb0beda0',
]
ALCF_TRANSFER_DEFAULT_TOKEN_FILE = Path.home() / '.globus' / 'alcf_transfer_tokens.json'

OLCF_TRANSFER_COLLECTIONS = [
    '5049be87-dfcb-4ea3-84d9-07b11647f5d2',
]
OLCF_TRANSFER_DEFAULT_TOKEN_FILE = Path.home() / '.globus' / 'olcf_transfer_tokens.json'
OLCF_REQUIRED_DOMAIN = 'opensso.ccs.ornl.gov'


class _AlcfDomainErrorHandler:
    def __call__(self, app, error):
        print(f"Encountered error '{error}', initiating login...")
        app.login(auth_params=ALCF_GA_PARAMS)


def _alcf_get_auth_object(force: bool = False):
    app = globus_sdk.UserApp(
        ALCF_APP_NAME,
        client_id=ALCF_AUTH_CLIENT_ID,
        scope_requirements={ALCF_SCOPE_CLIENT_ID: [ALCF_SCOPE_STRING]},
        config=globus_sdk.GlobusAppConfig(
            request_refresh_tokens=True,
            token_validation_error_handler=_AlcfDomainErrorHandler(),
        ),
    )
    if force:
        app.login(auth_params=ALCF_GA_PARAMS)
    return app.get_authorizer(ALCF_SCOPE_CLIENT_ID)


def alcf_authenticate() -> None:
    """Force an interactive ALCF login and store the resulting tokens."""
    _alcf_get_auth_object(force=True)


def get_alcf_iri_token(force_login: bool = False) -> str:
    """Return a valid ALCF IRI access token, refreshing silently if needed."""
    auth = _alcf_get_auth_object(force=force_login)
    auth.ensure_valid_token()
    return auth.access_token


def alcf_get_time_until_expiration(units: str = 'seconds') -> float:
    """Return the time until the ALCF access token expires."""
    auth = _alcf_get_auth_object(force=False)
    delta_t = auth.expires_at - time.time()
    if units == 'minutes':
        delta_t /= 60
    elif units == 'hours':
        delta_t /= 3600
    elif units != 'seconds':
        raise ValueError("units must be 'seconds', 'minutes', or 'hours'.")
    return round(delta_t, 2)


def _ensure_private_parent_dir(path: Path) -> None:
    # Tighten only a directory this script creates. --token-file is user-supplied, so an
    # unconditional chmod here would clamp an existing home or shared project directory to
    # 0700 and lock out its other users.
    try:
        path.parent.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        pass


def _load_tokens(token_file: Path) -> dict | None:
    if not token_file.exists():
        return None
    with token_file.open('r', encoding='utf-8') as f:
        return json.load(f)


def _save_tokens(token_file: Path, tokens: dict) -> None:
    _ensure_private_parent_dir(token_file)
    tmp = token_file.with_suffix('.tmp')
    # The temp path is predictable, so refuse to write through a pre-planted file or symlink.
    tmp.unlink(missing_ok=True)
    with os.fdopen(
        os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(tokens, f, indent=2)
    os.replace(tmp, token_file)
    os.chmod(token_file, stat.S_IRUSR | stat.S_IWUSR)


def _parse_scope_string(scope_string: str) -> set[str]:
    return set(scope_string.split()) if scope_string else set()


def _nersc_get_iri_token(token_response_data: dict) -> dict:
    for token_data in token_response_data.get('other_tokens', []):
        if NERSC_IRI_SCOPE in _parse_scope_string(token_data.get('scope', '')):
            return token_data
    raise RuntimeError(f'Missing token for required IRI scope: {NERSC_IRI_SCOPE}')


def _nersc_get_iri_token_refresh(stored: dict) -> str | None:
    try:
        return _nersc_get_iri_token(stored).get('refresh_token')
    except RuntimeError:
        return None


def _nersc_replace_iri_token(token_response_data: dict, iri_token_data: dict) -> dict:
    merged = dict(token_response_data)
    other_tokens = list(merged.get('other_tokens', []))
    for i, token_data in enumerate(other_tokens):
        if NERSC_IRI_SCOPE in _parse_scope_string(token_data.get('scope', '')):
            other_tokens[i] = iri_token_data
            break
    else:
        other_tokens.append(iri_token_data)
    merged['other_tokens'] = other_tokens
    return merged


def _nersc_validate_auth_data(auth_data: dict) -> dict:
    if auth_data.get('resource_server') != NERSC_RESOURCE_SERVER:
        raise RuntimeError(f'Missing token for required resource server: {NERSC_RESOURCE_SERVER}')
    granted = _parse_scope_string(auth_data.get('scope', ''))
    missing = NERSC_REQUIRED_SCOPES - granted
    if missing:
        raise RuntimeError(f'Missing required scopes: {sorted(missing)}')
    return _nersc_get_iri_token(auth_data)


def _nersc_refresh_token(client: globus_sdk.NativeAppAuthClient, refresh_token: str) -> dict | None:
    try:
        return client.oauth2_refresh_token(refresh_token).data
    except GlobusAPIError as exc:
        print(f'Refresh failed ({exc.http_status}); switching to interactive login.')
        return None


def _nersc_refresh_stored_tokens(
    client: globus_sdk.NativeAppAuthClient, stored: dict
) -> tuple[dict | None, bool]:
    iri_refresh = _nersc_get_iri_token_refresh(stored)
    if iri_refresh:
        iri_data = _nersc_refresh_token(client, iri_refresh)
        if iri_data is not None:
            return _nersc_replace_iri_token(stored, iri_data), True

    auth_refresh = stored.get('refresh_token') or (stored.get(NERSC_RESOURCE_SERVER) or {}).get(
        'refresh_token'
    )
    if auth_refresh:
        auth_data = _nersc_refresh_token(client, auth_refresh)
        if auth_data is not None:
            return auth_data, True

    return None, False


def _nersc_interactive_login(client: globus_sdk.NativeAppAuthClient) -> dict:
    client.oauth2_start_flow(
        requested_scopes=' '.join(sorted(NERSC_REQUESTED_SCOPES)),
        refresh_tokens=True,
    )
    print('Open this URL, login, and consent:')
    print(client.oauth2_get_authorize_url())
    code = input('\nEnter authorization code: ').strip()
    return client.oauth2_exchange_code_for_tokens(code).data


def get_nersc_iri_token(
    token_file: Path = NERSC_DEFAULT_TOKEN_FILE,
    force_login: bool = False,
    refresh_only: bool = False,
) -> str:
    """Return a valid NERSC IRI access token, refreshing or logging in as needed."""
    if force_login and refresh_only:
        raise ValueError('Choose only one of force_login or refresh_only.')

    client = globus_sdk.NativeAppAuthClient(NATIVE_APP_CLIENT_ID)
    auth_data = None
    used_refresh = False

    if not force_login:
        stored = _load_tokens(token_file)
        if stored:
            auth_data, used_refresh = _nersc_refresh_stored_tokens(client, stored)

    if auth_data is None:
        if refresh_only:
            raise RuntimeError('Refresh-only mode failed: no usable saved refresh token found.')
        auth_data = _nersc_interactive_login(client)

    try:
        iri_token_data = _nersc_validate_auth_data(auth_data)
    except RuntimeError as exc:
        if used_refresh and 'Missing token for required IRI scope' in str(exc):
            print('Refreshed tokens missing IRI scope; switching to interactive login.')
            auth_data = _nersc_interactive_login(client)
            iri_token_data = _nersc_validate_auth_data(auth_data)
        else:
            raise

    _save_tokens(token_file, auth_data)
    return iri_token_data['access_token']


def _nersc_transfer_refresh(client: globus_sdk.NativeAppAuthClient, stored: dict) -> dict | None:
    refresh_token = stored.get('refresh_token')
    if not refresh_token:
        return None
    try:
        response = client.oauth2_refresh_token(refresh_token)
        by_rs = response.by_resource_server
        if NERSC_AMSC_RESOURCE_SERVER in by_rs:
            return dict(by_rs[NERSC_AMSC_RESOURCE_SERVER])
        return dict(response.data)
    except GlobusAPIError as exc:
        print(
            f'Transfer token refresh failed ({exc.http_status}); falling back to interactive login.'
        )
        return None


def _nersc_transfer_interactive_login(client: globus_sdk.NativeAppAuthClient) -> dict:
    globus_scope = Scope(NERSC_TRANSFER_SCOPE_URL)
    data_access = [GCSCollectionScopes(c).data_access for c in NERSC_TRANSFER_COLLECTIONS]
    transfer_scope = TransferScopes.all.with_dependencies(data_access)
    globus_scope = globus_scope.with_dependency(transfer_scope)

    client.oauth2_start_flow(requested_scopes=str(globus_scope), refresh_tokens=True)
    print(f'Please go to this URL and login:\n{client.oauth2_get_authorize_url()}')
    auth_code = input('Enter the authorization code: ').strip()
    tokens = client.oauth2_exchange_code_for_tokens(auth_code)
    return dict(tokens.by_resource_server[NERSC_AMSC_RESOURCE_SERVER])


def _alcf_transfer_refresh(client: globus_sdk.NativeAppAuthClient, stored: dict) -> dict | None:
    refresh_token = stored.get('refresh_token')
    if not refresh_token:
        return None
    try:
        response = client.oauth2_refresh_token(refresh_token)
        by_rs = response.by_resource_server
        if NERSC_AMSC_RESOURCE_SERVER in by_rs:
            return dict(by_rs[NERSC_AMSC_RESOURCE_SERVER])
        return dict(response.data)
    except GlobusAPIError as exc:
        print(
            f'ALCF transfer token refresh failed ({exc.http_status}); falling back to interactive login.'
        )
        return None


def _alcf_transfer_interactive_login(client: globus_sdk.NativeAppAuthClient) -> dict:
    globus_scope = Scope(NERSC_TRANSFER_SCOPE_URL)
    data_access = [GCSCollectionScopes(c).data_access for c in ALCF_TRANSFER_COLLECTIONS]
    transfer_scope = TransferScopes.all.with_dependencies(data_access)
    globus_scope = globus_scope.with_dependency(transfer_scope)

    client.oauth2_start_flow(requested_scopes=str(globus_scope), refresh_tokens=True)
    print(f'Please go to this URL and login:\n{client.oauth2_get_authorize_url()}')
    auth_code = input('Enter the authorization code: ').strip()
    tokens = client.oauth2_exchange_code_for_tokens(auth_code)
    return dict(tokens.by_resource_server[NERSC_AMSC_RESOURCE_SERVER])


def get_alcf_transfer_token(
    token_file: Path = ALCF_TRANSFER_DEFAULT_TOKEN_FILE,
    force_login: bool = False,
) -> str:
    """Return a valid ALCF transfer access token, refreshing or logging in as needed."""
    client = globus_sdk.NativeAppAuthClient(NATIVE_APP_CLIENT_ID)

    stored = None
    if not force_login:
        stored = _load_tokens(token_file)
        if stored:
            stored = _alcf_transfer_refresh(client, stored)

    if stored is None:
        stored = _alcf_transfer_interactive_login(client)

    _save_tokens(token_file, stored)
    return stored['access_token']


def get_nersc_transfer_token(
    token_file: Path = NERSC_TRANSFER_DEFAULT_TOKEN_FILE,
    force_login: bool = False,
) -> str:
    """Return a valid NERSC transfer access token, refreshing or logging in as needed."""
    client = globus_sdk.NativeAppAuthClient(NATIVE_APP_CLIENT_ID)

    stored = None
    if not force_login:
        stored = _load_tokens(token_file)
        if stored:
            stored = _nersc_transfer_refresh(client, stored)

    if stored is None:
        stored = _nersc_transfer_interactive_login(client)

    _save_tokens(token_file, stored)
    return stored['access_token']


def _olcf_transfer_refresh(client: globus_sdk.NativeAppAuthClient, stored: dict) -> dict | None:
    refresh_token = stored.get('refresh_token')
    if not refresh_token:
        return None
    try:
        response = client.oauth2_refresh_token(refresh_token)
        by_rs = response.by_resource_server
        if NERSC_AMSC_RESOURCE_SERVER in by_rs:
            return dict(by_rs[NERSC_AMSC_RESOURCE_SERVER])
        return dict(response.data)
    except GlobusAPIError as exc:
        print(
            f'OLCF transfer token refresh failed ({exc.http_status}); falling back to interactive login.'
        )
        return None


def _olcf_transfer_interactive_login(client: globus_sdk.NativeAppAuthClient) -> dict:
    globus_scope = Scope(NERSC_TRANSFER_SCOPE_URL)
    data_access = [GCSCollectionScopes(c).data_access for c in OLCF_TRANSFER_COLLECTIONS]
    transfer_scope = TransferScopes.all.with_dependencies(data_access)
    globus_scope = globus_scope.with_dependency(transfer_scope)

    client.oauth2_start_flow(requested_scopes=str(globus_scope), refresh_tokens=True)
    authorize_url = client.oauth2_get_authorize_url(
        query_params={'session_required_single_domain': OLCF_REQUIRED_DOMAIN}
    )
    print(
        f'Please go to this URL and login with your ORNL ({OLCF_REQUIRED_DOMAIN}) '
        f'identity:\n{authorize_url}'
    )
    auth_code = input('Enter the authorization code: ').strip()
    tokens = client.oauth2_exchange_code_for_tokens(auth_code)
    return dict(tokens.by_resource_server[NERSC_AMSC_RESOURCE_SERVER])


def get_olcf_transfer_token(
    token_file: Path = OLCF_TRANSFER_DEFAULT_TOKEN_FILE,
    force_login: bool = False,
) -> str:
    """Return a valid OLCF transfer access token, refreshing or logging in as needed."""
    client = globus_sdk.NativeAppAuthClient(NATIVE_APP_CLIENT_ID)

    stored = None
    if not force_login:
        stored = _load_tokens(token_file)
        if stored:
            stored = _olcf_transfer_refresh(client, stored)

    if stored is None:
        stored = _olcf_transfer_interactive_login(client)

    _save_tokens(token_file, stored)
    return stored['access_token']
