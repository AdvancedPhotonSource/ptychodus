from collections.abc import Mapping, MutableMapping
from importlib.metadata import version
from pprint import pformat
from typing import Final, Union
import logging

from globus_sdk import AccessTokenAuthorizer, NativeAppAuthClient, RefreshTokenAuthorizer

from ...api.observer import Observable

logger = logging.getLogger(__name__)

AuthorizerTypes = Union[AccessTokenAuthorizer, RefreshTokenAuthorizer]
ScopeAuthorizerMapping = Mapping[str, AuthorizerTypes]
ScopeAuthorizerMutableMapping = Mapping[str, AuthorizerTypes]


class GlobusAuthorizationService(Observable):
    CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self) -> None:
        logger.info('\tGlobus SDK ' + version('globus-sdk'))

        super().__init__()
        self._authClient = NativeAppAuthClient(GlobusAuthorizationService.CLIENT_ID)

    def reauthorize(self, requestedScopes: list[str]) -> None:
        logger.debug('Requested Scopes: {pformat(requestedScopes)}')
        self._authClient.oauth2_start_flow(requested_scopes=requestedScopes, refresh_tokens=True)
        self.notifyObservers()

    def getAuthorizeURL(self) -> str:
        authorizeURL = self._authClient.oauth2_get_authorize_url()
        logger.debug(f'Authorize URL: {authorizeURL}')
        return authorizeURL

    def getAuthorizers(self, authCode: str) -> ScopeAuthorizerMapping:
        tokenResponse = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        logger.debug(f'Token response: {tokenResponse}')
        authorizers = dict()  # TODO typing

        for resourceServer, tokenData in tokenResponse.by_resource_server.items():
            authorizer = RefreshTokenAuthorizer(
                refresh_token=tokenData['refresh_token'],
                auth_client=self._authClient,
                access_token=tokenData['access_token'],
                expires_at=tokenData['expires_at_seconds'],
            )

            scope = tokenData['scope']
            authorizers[scope] = authorizer

        logger.debug('Authorizers: {pformat(authorizers)}')

        return authorizers
