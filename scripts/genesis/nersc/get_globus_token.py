#!/usr/bin/env python3
import argparse
import json
import os
import stat
import time
import urllib.error
import urllib.request
from pathlib import Path

import globus_sdk
from globus_sdk.exc import GlobusAPIError

CLIENT_ID = 'fae5c579-490a-4d76-b6eb-d78f65caeb63'
RESOURCE_SERVER = 'auth.globus.org'
FACILITY_SCOPE_MAP = {
    'nersc': {
        'scope': ('https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api'),
        'label': 'NERSC IRI API',
    },
    'alcf': {
        'scope': ('https://auth.globus.org/scopes/6be511f6-a071-471f-9bc0-02a0d0836723/filesystem'),
        'label': 'ALCF IRI API',
    },
}
NERSC_IRI_SCOPE = FACILITY_SCOPE_MAP['nersc']['scope']
ALCF_IRI_SCOPE = FACILITY_SCOPE_MAP['alcf']['scope']
DEFAULT_FACILITIES = tuple(FACILITY_SCOPE_MAP)
REQUIRED_SCOPES = {
    'openid',
    'profile',
    'email',
    'urn:globus:auth:scope:auth.globus.org:view_identities',
}
DEFAULT_IRI_VALIDATE_URL = 'https://api.iri.nersc.gov/api/v1/account/projects'

SCOPE_LABELS = {config['scope']: config['label'] for config in FACILITY_SCOPE_MAP.values()}


def parse_args() -> argparse.Namespace:
    default_token_file = Path.home() / '.globus' / 'auth_tokens.json'
    parser = argparse.ArgumentParser(
        description=(
            'Get Globus Auth tokens with required scopes. '
            'Tokens are saved to a secure local file by default.'
        )
    )
    parser.add_argument(
        '--token-file',
        type=Path,
        default=default_token_file,
        help=f'Path for saved token JSON (default: {default_token_file})',
    )
    parser.add_argument(
        '--print-token',
        action='store_true',
        help='Print the access token to stdout (off by default).',
    )
    parser.add_argument(
        '--facilities',
        nargs='+',
        choices=sorted(FACILITY_SCOPE_MAP),
        default=list(DEFAULT_FACILITIES),
        help=(f'Facility tokens to request and manage (default: {" ".join(DEFAULT_FACILITIES)})'),
    )
    parser.add_argument(
        '--force-login',
        action='store_true',
        help='Skip refresh and force interactive browser login.',
    )
    parser.add_argument(
        '--refresh-only',
        action='store_true',
        help='Refresh saved tokens only; do not fall back to interactive login.',
    )
    parser.add_argument(
        '--prompt-login',
        action='store_true',
        help='Add prompt=login to the Globus authorize URL to force re-authentication.',
    )
    parser.add_argument(
        '--validate-iri',
        action='store_true',
        help='Validate the IRI token by calling the IRI account/projects endpoint.',
    )
    parser.add_argument(
        '--iri-validate-url',
        default=DEFAULT_IRI_VALIDATE_URL,
        help=(f'IRI endpoint used by --validate-iri (default: {DEFAULT_IRI_VALIDATE_URL})'),
    )
    return parser.parse_args()


def get_selected_facilities(args: argparse.Namespace) -> list[str]:
    # Deduplicate while preserving CLI order.
    return list(dict.fromkeys(args.facilities))


def get_required_other_scopes(facilities: list[str]) -> set[str]:
    return {FACILITY_SCOPE_MAP[facility]['scope'] for facility in facilities}


def get_requested_scopes(facilities: list[str]) -> set[str]:
    return REQUIRED_SCOPES | get_required_other_scopes(facilities)


def parse_scope_string(scope_string: str) -> set[str]:
    return set(scope_string.split()) if scope_string else set()


def ensure_private_parent_dir(path: Path) -> None:
    # Tighten only a directory this script creates. --token-file is user-supplied, so an
    # unconditional chmod here would clamp an existing home or shared project directory to
    # 0700 and lock out its other users.
    try:
        path.parent.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        pass


def load_tokens(token_file: Path) -> dict | None:
    if not token_file.exists():
        return None
    with token_file.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_tokens(token_file: Path, tokens: dict) -> None:
    ensure_private_parent_dir(token_file)
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


def get_refresh_token(stored_tokens: dict) -> str | None:
    if 'refresh_token' in stored_tokens:
        return stored_tokens.get('refresh_token')

    auth_tokens = stored_tokens.get(RESOURCE_SERVER)
    if isinstance(auth_tokens, dict):
        return auth_tokens.get('refresh_token')

    return None


def get_token_for_scope(token_response_data: dict, scope: str, *, label: str = 'auxiliary') -> dict:
    for token_data in token_response_data.get('other_tokens', []):
        if scope in parse_scope_string(token_data.get('scope', '')):
            return token_data
    raise RuntimeError(f'Missing token for required {label} scope: {scope}')


def get_facility_token(token_response_data: dict, facility: str) -> dict:
    scope = FACILITY_SCOPE_MAP[facility]['scope']
    label = FACILITY_SCOPE_MAP[facility]['label']
    return get_token_for_scope(token_response_data, scope, label=label)


def get_refresh_token_for_scope(stored_tokens: dict, scope: str) -> str | None:
    try:
        return get_token_for_scope(
            stored_tokens,
            scope,
            label=SCOPE_LABELS.get(scope, 'auxiliary'),
        ).get('refresh_token')
    except RuntimeError:
        return None


def replace_token_for_scope(
    token_response_data: dict, scope: str, refreshed_token_data: dict
) -> dict:
    merged = dict(token_response_data)
    other_tokens = list(merged.get('other_tokens', []))
    for index, token_data in enumerate(other_tokens):
        if scope in parse_scope_string(token_data.get('scope', '')):
            other_tokens[index] = refreshed_token_data
            break
    else:
        other_tokens.append(refreshed_token_data)
    merged['other_tokens'] = other_tokens
    return merged


def merge_auth_token_data(token_response_data: dict, refreshed_auth_data: dict) -> dict:
    merged = dict(refreshed_auth_data)
    merged['other_tokens'] = list(token_response_data.get('other_tokens', []))
    return merged


def validate_auth_data(auth_data: dict, facilities: list[str]) -> dict:
    if auth_data.get('resource_server') != RESOURCE_SERVER:
        raise RuntimeError(f'Missing token for required resource server: {RESOURCE_SERVER}')

    granted = parse_scope_string(auth_data.get('scope', ''))
    missing = REQUIRED_SCOPES - granted
    if missing:
        raise RuntimeError(f'Missing required scopes: {sorted(missing)}')

    for facility in facilities:
        get_facility_token(auth_data, facility)

    return auth_data


def validate_nersc_iri_token(nersc_iri_token_data: dict, validate_url: str) -> dict | list:
    request = urllib.request.Request(
        validate_url,
        headers={
            'accept': 'application/json',
            'Authorization': f'Bearer {nersc_iri_token_data["access_token"]}',
        },
        method='GET',
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode('utf-8')
            data = json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode('utf-8')
        details = body.strip() or exc.reason
        raise RuntimeError(
            f'IRI validation failed with HTTP {exc.code} from {validate_url}: {details}'
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f'IRI validation request failed for {validate_url}: {exc.reason}'
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f'IRI validation returned non-JSON data from {validate_url}') from exc

    if isinstance(data, dict):
        session_info = data.get('session_info')
        if isinstance(session_info, dict):
            authentications = session_info.get('authentications')
            if isinstance(authentications, dict) and not authentications:
                raise RuntimeError(
                    'IRI validation succeeded but session_info.authentications is empty. '
                    'Re-run with --force-login --prompt-login and use a Chrome incognito window.'
                )

    return data


def interactive_login(
    client: globus_sdk.NativeAppAuthClient,
    facilities: list[str],
    *,
    prompt_login: bool = False,
) -> dict:
    client.oauth2_start_flow(
        requested_scopes=' '.join(sorted(get_requested_scopes(facilities))),
        refresh_tokens=True,
    )
    print('Open this URL, login, and consent:')
    if prompt_login:
        print(client.oauth2_get_authorize_url(prompt='login'))
    else:
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
    return token_response.data


def refresh_tokens(client: globus_sdk.NativeAppAuthClient, refresh_token: str) -> dict | None:
    try:
        token_response = client.oauth2_refresh_token(refresh_token)
        return token_response.data
    except GlobusAPIError as exc:
        print(f'Refresh failed ({exc.http_status}); switching to interactive login.')
        return None


def refresh_stored_tokens(
    client: globus_sdk.NativeAppAuthClient,
    stored_tokens: dict,
    facilities: list[str],
) -> tuple[dict | None, bool]:
    refreshed_tokens = dict(stored_tokens)
    used_refresh = False
    auth_refresh_token = get_refresh_token(stored_tokens)
    if auth_refresh_token:
        auth_data = refresh_tokens(client, auth_refresh_token)
        if auth_data is not None:
            refreshed_tokens = merge_auth_token_data(refreshed_tokens, auth_data)
            used_refresh = True

    for facility in facilities:
        scope = FACILITY_SCOPE_MAP[facility]['scope']
        refresh_token = get_refresh_token_for_scope(stored_tokens, scope)
        if refresh_token:
            refreshed_token_data = refresh_tokens(client, refresh_token)
            if refreshed_token_data is not None:
                refreshed_tokens = replace_token_for_scope(
                    refreshed_tokens, scope, refreshed_token_data
                )
                used_refresh = True

        try:
            get_facility_token(refreshed_tokens, facility)
        except RuntimeError:
            return None, used_refresh

    if used_refresh:
        return refreshed_tokens, True

    return None, False


def main() -> None:
    args = parse_args()
    if args.force_login and args.refresh_only:
        raise RuntimeError('Choose only one of --force-login or --refresh-only')
    facilities = get_selected_facilities(args)
    if args.validate_iri and 'nersc' not in facilities:
        raise RuntimeError("--validate-iri requires including the 'nersc' facility")

    client = globus_sdk.NativeAppAuthClient(CLIENT_ID)

    auth_data = None
    used_refresh = False
    if not args.force_login:
        stored = load_tokens(args.token_file)
        if stored:
            auth_data, used_refresh = refresh_stored_tokens(client, stored, facilities)

    if auth_data is None:
        if args.refresh_only:
            facility_labels = ', '.join(
                FACILITY_SCOPE_MAP[facility]['label'] for facility in facilities
            )
            raise RuntimeError(
                'Refresh-only mode failed. No usable saved refresh token was found '
                f'or token refresh did not return all required tokens for: {facility_labels}.'
            )
        auth_data = interactive_login(client, facilities, prompt_login=args.prompt_login)

    try:
        validate_auth_data(auth_data, facilities)
    except RuntimeError as exc:
        if used_refresh and 'Missing token for required ' in str(exc):
            print(
                'Refreshed tokens did not include all required facility tokens; '
                'switching to interactive login.'
            )
            auth_data = interactive_login(client, facilities, prompt_login=args.prompt_login)
            validate_auth_data(auth_data, facilities)
        else:
            raise

    save_tokens(args.token_file, auth_data)

    if args.validate_iri:
        nersc_iri_token_data = get_facility_token(auth_data, 'nersc')
        validation_data = validate_nersc_iri_token(nersc_iri_token_data, args.iri_validate_url)
        print(f'IRI validation succeeded against {args.iri_validate_url}')
        if isinstance(validation_data, dict):
            session_info = validation_data.get('session_info')
            if isinstance(session_info, dict):
                session_id = session_info.get('session_id')
                if session_id:
                    print(f'IRI session_id: {session_id}')
        elif isinstance(validation_data, list):
            print(f'IRI validation response items: {len(validation_data)}')

    print(f'Saved token data to {args.token_file}')
    print(f'Selected facilities: {", ".join(facilities)}')
    print(f'Granted Globus Auth scopes: {auth_data.get("scope", "")}')
    token_data_by_facility = {
        facility: get_facility_token(auth_data, facility) for facility in facilities
    }
    for facility in facilities:
        label = FACILITY_SCOPE_MAP[facility]['label']
        token_data = token_data_by_facility[facility]
        expires_at = token_data.get('expires_at_seconds')
        if expires_at:
            ttl = int(expires_at - time.time())
            print(f'{label} access token valid for ~{max(ttl, 0)} seconds.')
        print(f'{label} token resource server: {token_data.get("resource_server")}')
        print(f'{label} token scopes: {token_data.get("scope", "")}')

    if args.print_token:
        for facility in facilities:
            label = FACILITY_SCOPE_MAP[facility]['label']
            print(f'\n{label} access token:')
            print(token_data_by_facility[facility]['access_token'])
    else:
        print('Selected facility access tokens not printed (use --print-token to display them).')


if __name__ == '__main__':
    main()
