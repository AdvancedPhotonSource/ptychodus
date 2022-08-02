from typing import Any
import logging

from globus_automate_client import FlowsClient
from globus_automate_client.flows_client import RUN_FLOWS_SCOPE, VIEW_FLOWS_SCOPE
from globus_sdk import NativeAppAuthClient, RefreshTokenAuthorizer

from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


class WorkflowClient:
    CLIENT_ID: str = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self, settings: WorkflowSettings, authorizer: RefreshTokenAuthorizer) -> None:
        self._settings = settings
        self._authorizer = authorizer
        self._client = FlowsClient.new_client(client_id=WorkflowClient.CLIENT_ID,
                                              authorizer=authorizer,
                                              authorizer_callback=self._authorizerRetriever)

    def _authorizerRetriever(self, flowUrl: str, flowScope: str,
                             clientId: str) -> RefreshTokenAuthorizer:
        return self._authorizer

    def printFlows(self) -> None:
        myFlows = self._client.list_flows()
        logger.info(myFlows)

    def runFlow(self) -> None:
        flowID = str(self._settings.flowID.value)
        flowScope = None
        flowInput: dict[Any, Any] = dict()
        self._client.run_flow(flowID, flowScope, flowInput)


class WorkflowClientBuilder:

    def __init__(self, settings: WorkflowSettings) -> None:
        self._settings = settings
        self._authClient = NativeAppAuthClient(WorkflowClient.CLIENT_ID)

    def getAuthorizeUrl(self) -> str:
        # TODO Add deployed flow scope:
        #     https://auth.globus.org/scopes/904d62f5-139e-45e1-a125-eaf69bf1fb68/flow_904d62f5_139e_45e1_a125_eaf69bf1fb68_user%22
        # You can fetch the scope for a flow using the ID using:
        #     globus-automate flow get 904d62f5-139e-45e1-a125-eaf69bf1fb68 | grep scope
        requestedScopes = [RUN_FLOWS_SCOPE, VIEW_FLOWS_SCOPE]
        self._authClient.oauth2_start_flow(requested_scopes=requestedScopes, refresh_tokens=True)
        return self._authClient.oauth2_get_authorize_url()

    def build(self, authCode: str) -> FlowsClient:
        tokens = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        logger.debug(tokens)
        flowsTokens = tokens.by_resource_server['flows.globus.org']
        authorizer = RefreshTokenAuthorizer(flowsTokens['refresh_token'], self._authClient)
        return WorkflowClient(self._settings, authorizer)
