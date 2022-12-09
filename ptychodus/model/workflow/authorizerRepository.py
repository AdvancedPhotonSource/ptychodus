from collections.abc import Mapping
from importlib.metadata import version
from pprint import pformat
from typing import Final, Union
import logging
import threading

from globus_sdk import AccessTokenAuthorizer, NativeAppAuthClient, RefreshTokenAuthorizer

from .api import WorkflowAuthorizerRepository

logger = logging.getLogger(__name__)

AuthorizerTypes = Union[AccessTokenAuthorizer, RefreshTokenAuthorizer]
ScopeAuthorizerMapping = Mapping[str, AuthorizerTypes]


class GlobusAuthorizerRepository(WorkflowAuthorizerRepository):
    CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self) -> None:
        super().__init__()
        logger.info('\tGlobus SDK ' + version('globus-sdk'))
        # TODO handle auth cancelled
        # TODO secure tokens for future invocations
        self._authClient = NativeAppAuthClient(GlobusAuthorizerRepository.CLIENT_ID)
        self._authorizers: dict[str, AuthorizerTypes] = dict()
        self.isAuthorizedEvent = threading.Event()

    def requestAuthorization(self, scopes: list[str]) -> None:
        self.isAuthorizedEvent.clear()
        logger.debug('Requested scopes: {pformat(scopes)}')
        self._authClient.oauth2_start_flow(requested_scopes=scopes, refresh_tokens=True)

    @property
    def isAuthorized(self) -> bool:
        return self.isAuthorizedEvent.is_set()

    def getAuthorizeURL(self) -> str:
        authorizeURL = self._authClient.oauth2_get_authorize_url()
        logger.debug(f'Authorize URL: {authorizeURL}')
        return authorizeURL

    def setCodeFromAuthorizeURL(self, code: str) -> None:
        logger.debug(f'Authorize code: {code}')
        tokenResponse = self._authClient.oauth2_exchange_code_for_tokens(code.strip())
        logger.debug(f'Token response: {tokenResponse}')

        for resourceServer, tokenData in tokenResponse.by_resource_server.items():
            authorizer = RefreshTokenAuthorizer(
                refresh_token=tokenData['refresh_token'],
                auth_client=self._authClient,
                access_token=tokenData['access_token'],
                expires_at=tokenData['expires_at_seconds'],
            )

            scope = tokenData['scope']
            logger.debug(f'Scope={scope} Authorizer={authorizer}')
            self._authorizers[scope] = authorizer

        self.isAuthorizedEvent.set()

    def getAuthorizers(self) -> ScopeAuthorizerMapping:
        return self._authorizers
