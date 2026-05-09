import globus_sdk
from globus_sdk.login_flows import LocalServerLoginFlowManager # Needed to access globus_sdk.gare
import os.path
import time

# Globus UserApp name
APP_NAME = "alcf_facility_api_app"

# Public auth client
AUTH_CLIENT_ID = "8b84fc2d-49e9-49ea-b54d-b3a29a70cf31"

# Globus scope
SCOPE_CLIENT_ID = "6be511f6-a071-471f-9bc0-02a0d0836723"
SCOPE_STRING = f"https://auth.globus.org/scopes/{SCOPE_CLIENT_ID}/filesystem"

# Path where access and refresh tokens are stored
TOKENS_PATH = f"{os.path.expanduser('~')}/.globus/app/{AUTH_CLIENT_ID}/{APP_NAME}/tokens.json"

# Globus authorizer parameters to point to specific identity providers
GA_PARAMS = globus_sdk.gare.GlobusAuthorizationParameters(session_required_policies=["a128e981-c9a5-417a-97ab-8571c9831bff"])


# Error handler to guide user through specific identity providers 
class DomainBasedErrorHandler:
    def __call__(self, app, error):
        print(f"Encountered error '{error}', initiating login...")
        app.login(auth_params=GA_PARAMS)


# Get refresh authorizer object
def get_auth_object(force=False):
    """
    Create a Globus UserApp with the Facility API scope
    and trigger the authentication process. If authentication
    has already happened, existing tokens will be reused.
    """

    # Create Globus user application
    app = globus_sdk.UserApp(
        APP_NAME,
        client_id=AUTH_CLIENT_ID,
        scope_requirements={SCOPE_CLIENT_ID: [SCOPE_STRING]},
        config=globus_sdk.GlobusAppConfig(
            request_refresh_tokens=True,
            token_validation_error_handler=DomainBasedErrorHandler()
        ),
    )

    # Force re-login if required
    if force:
        app.login(auth_params=GA_PARAMS)

    # Authenticate using your Globus account or reuse existing tokens
    auth = app.get_authorizer(SCOPE_CLIENT_ID)

    # Return the Globus refresh token authorizer
    return auth


# Get access token
def get_access_token():
    """
    Load existing tokens, refresh the access token if necessary,
    and return the valid access token. If there is no token stored
    in the home directory, or if the refresh token is expired following
    6 months of inactivity, an authentication will be triggered.
    """

    # Get authorizer object and authenticate if need be
    auth = get_auth_object(force=False)

    # Make sure the stored access token if valid, and refresh otherwise
    auth.ensure_valid_token()

    # Return the access token
    return auth.access_token


# Get time until token expiration
def get_time_until_token_expiration(units="seconds"):
    """
    Returns the time until the access token expires, in units of
    seconds, minutes, or hours. Negative times reveal that the token
    is expired already.
    """

    # Get authorizer object
    auth = get_auth_object(force=False)

    # Gather the time difference between now and the expiration time (both Unix timestamps)
    now = time.time()
    delta_t = auth.expires_at - now

    # Convert units
    if units == "seconds":
        delta_t = delta_t
    elif units == "minutes":
        delta_t = delta_t / 60
    elif units == "hours":
        delta_t = delta_t / 3600
    else:
        return "Error: units must be 'seconds', 'minutes', or 'hours'."
    
    # Return the time difference in the requested Units
    return round(delta_t, 2)


# If this file is executed as a script ...
if __name__ == "__main__":

    # Imports
    import argparse

    # Exception to raise in case of errors
    class FacilityAPIAuthError(Exception):
        pass

    # Constant
    AUTHENTICATE_ACTION = "authenticate"
    GET_ACCESS_TOKEN_ACTION = "get_access_token"
    GET_TOKEN_EXPIRATION_ACTION = "get_time_until_token_expiration"

    # Define possible arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=[AUTHENTICATE_ACTION, GET_ACCESS_TOKEN_ACTION, GET_TOKEN_EXPIRATION_ACTION])
    parser.add_argument("--units", choices=['seconds', 'minutes', 'hours'], default='seconds', help="Units for the time until token expiration")
    args = parser.parse_args()

    # Authentication
    if args.action == AUTHENTICATE_ACTION:
        
        # Authenticate using Globus account
        _ = get_auth_object(force=True)

    # Get token
    elif args.action == GET_ACCESS_TOKEN_ACTION:

        # Make sure tokens exist
        # This is important otherwise the CLI will print more than just the access token
        if not os.path.isfile(TOKENS_PATH):
            raise FacilityAPIAuthError('Access token does not exist. '
                'Please authenticate by running "python3 globus_access_token.py authenticate".')

        # Load tokens, refresh token if necessary, and print access token
        print(get_access_token())

    # Get token expiration
    elif args.action == GET_TOKEN_EXPIRATION_ACTION:

        # Make sure tokens exist
        # This is important otherwise the CLI will print more than just the access token
        if not os.path.isfile(TOKENS_PATH):
            raise FacilityAPIAuthError('Access token does not exist. '
                'Please authenticate by running "python3 globus_access_token.py authenticate".')
        
        print(get_time_until_token_expiration(args.units))