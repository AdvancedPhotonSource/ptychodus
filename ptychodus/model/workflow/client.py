import logging

from globus_automate_client import FlowsClient
from globus_sdk import NativeAppAuthClient, RefreshTokenAuthorizer

from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


class WorkflowClient:
    CLIENT_ID: str = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self, settings: WorkflowSettings, authorizer: RefreshTokenAuthorizer) -> None:
        self._settings = settings
        self._authorizer = authorizer
        self._client = FlowsClient.new_client(
                client_id=WorkflowClient.CLIENT_ID,
                authorizer_callback=self._authorizerRetriever)

    def _authorizerRetriever(self, flowUrl: str, flowScope: str,
                             clientId: str) -> RefreshTokenAuthorizer:
        return self._authorizer

    def printFlows(self) -> None:
        myFlows = self._client.list_flows()
        logger.info(myFlows)

    def run(self) -> None:
        flowID = str(self._settings.flowID.value)
        flowScope = None
        flowInput = dict()
        self._client.run_flow(flowID, flowScope, flowInput)


class WorkflowClientBuilder:
    def __init__(self, settings: WorkflowSettings) -> None:
        self._settings = settings
        self._authClient = NativeAppAuthClient(WorkflowClient.CLIENT_ID)
        # FIXME MAYBE? requested_scopes: list[str], refresh_tokens:bool=True
        self._authClient.oauth2_start_flow()

    def getAuthorizeUrl(self) -> str:
        return self._authClient.oauth2_get_authorize_url()

    def build(self, authCode: str) -> FlowsClient:
        tokens = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        flowsRefreshToken = '' # FIXME tokens['?']
        authorizer = RefreshTokenAuthorizer(flowsRefreshToken, self._authClient)
        return WorkflowClient(self._settings, authorizer)
