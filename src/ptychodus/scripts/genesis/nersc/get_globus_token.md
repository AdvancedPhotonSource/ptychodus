# `get_globus_token.py` Usage

This document explains how to use:

`get_globus_token.py`

## What it does

- Gets tokens for both the NERSC IRI API and the ALCF IRI API by default.
- Supports `--facilities` to request tokens for only the listed facilities.
- Requests a Globus Auth token with these required scopes:
  - `openid`
  - `profile`
  - `email`
  - `urn:globus:auth:scope:auth.globus.org:view_identities`
  - NERSC IRI token: `https://auth.globus.org/scopes/ed3e577d-f7f3-4639-b96e-ff5a8445d699/iri_api`
  - ALCF IRI token: `https://auth.globus.org/scopes/6be511f6-a071-471f-9bc0-02a0d0836723/filesystem`
- Preserves the full token response, including `other_tokens`.
- Extracts the NERSC IRI access token from `other_tokens`.
- Requires tokens for the selected facilities to be present in `other_tokens`.
- Prints both the NERSC IRI access token and the ALCF IRI access token with `--print-token`.
- Can optionally validate the NERSC IRI token by calling the IRI `account/projects` endpoint.
- Saves token data to a secure local file by default.
- Reuses and refreshes saved tokens when possible.

## Prerequisites

1. Python 3.9+ (recommended).
2. `globus-sdk` installed:

```bash
pip install globus-sdk
```

## Default behavior (recommended)

Run:

```bash
python get_globus_token.py
```

What happens:

1. If a refresh token exists in the token file, the script tries to refresh automatically.
2. If refresh is not possible, it starts interactive login and prints a Globus URL to obtain both the NERSC IRI API token and the ALCF IRI API token.
3. After you authorize, paste the returned auth code in the terminal.
4. Token JSON is saved to:
   - `~/.globus/auth_tokens.json`
5. The NERSC IRI and ALCF IRI access tokens from `other_tokens` are **not** printed by default.

## Select facilities

By default, the script requests tokens for both facilities:

```bash
python get_globus_token.py --facilities nersc alcf
```

You can limit token acquisition to a subset:

```bash
python get_globus_token.py --facilities nersc
python get_globus_token.py --facilities alcf
```

Supported facility names are:

- `nersc`
- `alcf`

## Print token to terminal (optional)

```bash
python get_globus_token.py --print-token
```

Use this only when needed, since terminal logs/history may expose tokens.
The printed tokens correspond to the selected facilities. By default that means the NERSC IRI API token and the ALCF IRI API token.

## Force a new interactive login

```bash
python get_globus_token.py --force-login
```

This skips refresh and always performs browser auth.

## Force a fresh IdP login prompt

```bash
python get_globus_token.py --force-login --prompt-login
```

This adds `prompt=login` to the Globus authorization URL so Globus forces a fresh login at the identity provider instead of reusing an existing browser session.
This is useful when the server side shows an empty `session_info.authentications` object or when the IRI API returns `401` with a token that otherwise looks valid.

## Refresh saved tokens only

```bash
python get_globus_token.py --refresh-only
```

This attempts to refresh saved tokens without opening a browser login flow.
It refreshes the top-level Globus Auth token when possible and also refreshes the selected facility tokens from `other_tokens` when their refresh tokens are available.
If refresh is not possible, or if refresh does not return all requested facility tokens, the script exits with an error instead of starting interactive login.

## Validate the NERSC IRI token

```bash
python get_globus_token.py --validate-iri
```

This calls:

```bash
GET https://api.iri.nersc.gov/api/v1/account/projects
```

with the NERSC IRI access token from `other_tokens`.
If the response includes `session_info.authentications: {}`, the script treats that as a bad session and exits with guidance to re-run using `--force-login --prompt-login`.

`--validate-iri` requires that `nersc` is included in `--facilities`.

You can combine validation with token printing or refresh-only mode:

```bash
python get_globus_token.py --refresh-only --validate-iri --print-token
python get_globus_token.py --force-login --prompt-login --validate-iri
python get_globus_token.py --facilities nersc --validate-iri --print-token
```

## Use a custom token file path

```bash
python get_globus_token.py --token-file /path/to/auth_tokens.json
```

The script writes with private permissions (`0600`) and sets parent directory permissions to `0700`.

## Common troubleshooting

- `ModuleNotFoundError: No module named 'globus_sdk'`
  - Install dependency: `pip install globus-sdk`
- Refresh fails and script asks for login again
  - This is expected when refresh token expired/revoked.
- Refresh-only mode fails
  - Re-run without `--refresh-only` to allow interactive login, or use `--force-login` to start a fresh browser auth flow.
- Authorization code exchange fails with `invalid_grant`
  - Re-run the script and paste a fresh authorization code from Globus. This can happen if the code was empty, expired, or already used.
- `Missing required scopes`
  - Re-run with `--force-login` and ensure consent is granted for requested scopes.
- `Missing token for required NERSC IRI API scope`
  - Re-run with `--force-login` and ensure consent is granted for the NERSC IRI scope.
- `Missing token for required ALCF IRI API scope`
  - Re-run with `--force-login` and ensure consent is granted for the ALCF IRI scope.
- IRI API returns `401`
  - Re-run the script with `--force-login --prompt-login` and open the authorization URL in a Chrome incognito window before completing login.
- Server shows `session_info.authentications: {}`
  - Re-run the script with `--force-login --prompt-login` so Globus forces a fresh identity-provider login instead of reusing an existing browser session.
