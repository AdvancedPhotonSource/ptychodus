# mypy: ignore-errors
"""
This script allows a user to login via Globus Auth to the data movement API
It is equivalent to any third party Globus App that requests the DEFAULT_SCOPE below and calls
into the API. Users may also run the 'login' command to get an access token they can paste into the web UI
or use with curl to authenticate to the API.
"""

from typing import List
import typer
import globus_sdk
from globus_sdk.scopes import TransferScopes
from globus_sdk.gare import GlobusAuthorizationParameters
import requests
import json


CLIENT_ID = '57554043-6dd8-410b-8ece-cd54e9c003bb'
DEFAULT_SCOPE = 'https://auth.globus.org/scopes/08da012c-6998-46f9-9375-a6985ebe3f2b/transfer'
RESOURCE_SERVER = '08da012c-6998-46f9-9375-a6985ebe3f2b'
globus_app = globus_sdk.UserApp(
    'amsc-client-login',
    client_id=CLIENT_ID,
    scope_requirements={RESOURCE_SERVER: DEFAULT_SCOPE},
)

app = typer.Typer(no_args_is_help=True)


@app.command()
def login(mapped_collections: List[str] = None, domain: str = None):
    """
    Login to Globus and get a token based for the required Collections and Domains.
    For instance:
    python generate_token.py login \
        --mapped-collections 9d6d994a-6d04-11e5-ba46-22000b92c6ec \
        --domain opensso.ccs.ornl.gov
    """
    scope = globus_sdk.Scope(DEFAULT_SCOPE)
    params = {}

    if mapped_collections:
        data_access = [
            globus_sdk.scopes.GCSCollectionScopes(mc).data_access for mc in mapped_collections
        ]
        transfer_scope = TransferScopes.all.with_dependencies(data_access)
    else:
        transfer_scope = TransferScopes.all
    scope = scope.with_dependency(transfer_scope)
    print(f'Logging in with scope: {scope}')
    globus_app.add_scope_requirements({RESOURCE_SERVER: scope})

    if domain:
        params = GlobusAuthorizationParameters(session_required_single_domain=[domain])

    globus_app.login(auth_params=params)
    print('Login Successful. Access token is:\n\n')

    print(globus_app.get_authorizer(RESOURCE_SERVER).access_token)


@app.command()
def logout():
    """
    Logout from the service
    """
    globus_app.logout()
    print('You have been logged out.')


@app.command()
def token():
    """
    Print your Globus Bearer token for use with the API endpoints
    """
    app = globus_sdk.UserApp('amsc-client-login', client_id=CLIENT_ID)
    print(app.get_authorizer(RESOURCE_SERVER).access_token)


@app.command()
def test_transfer(source_url: str = None, destination_url: str = None):
    auth = {'Authorization': f'Bearer {globus_app.get_authorizer(RESOURCE_SERVER).access_token}'}
    if source_url and destination_url:
        payload = {
            'source_url': source_url,
            'destination_url': destination_url,
            'label': 'Test Transfer on Tutorial Endpoints',
        }
    else:
        payload = {
            'source_url': 'globus://6c54cade-bde5-45c1-bdea-f4bd71dba2cc/home/share/godata/',
            'destination_url': 'globus://31ce9ba0-176d-45a5-add3-f37d233ba47d/~/godata',
            'label': 'Test Transfer on Tutorial Endpoints',
        }

    r = requests.post('https://amsc-data-api.nersc.gov/transfer/globus', json=payload, headers=auth)
    print(json.dumps(r.json()))


@app.command()
def check_transfer(transfer_uuid: str):
    auth = {'Authorization': f'Bearer {globus_app.get_authorizer(RESOURCE_SERVER).access_token}'}
    r = requests.get(
        f'https://amsc-data-api.nersc.gov/transfer/globus/{transfer_uuid}', headers=auth
    )
    print(json.dumps(r.json()))


if __name__ == '__main__':
    app()
