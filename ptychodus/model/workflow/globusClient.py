from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Optional
from uuid import UUID
import json
import logging

from globus_automate_client import FlowsClient
from globus_automate_client.flows_client import (MANAGE_FLOWS_SCOPE, RUN_FLOWS_SCOPE,
                                                 RUN_STATUS_SCOPE, VIEW_FLOWS_SCOPE)
from globus_sdk import NativeAppAuthClient, OAuthTokenResponse, RefreshTokenAuthorizer

from .client import WorkflowClient, WorkflowClientBuilder, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


class GlobusWorkflowClient(WorkflowClient):
    CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self, settings: WorkflowSettings, authClient: NativeAppAuthClient,
                 tokenResponse: OAuthTokenResponse) -> None:
        super().__init__()
        self._settings = settings
        self._tokenResponse = tokenResponse
        self._authorizerMapping = {
            resourceServer: RefreshTokenAuthorizer(tokenData['refresh_token'], authClient)
            for resourceServer, tokenData in tokenResponse.by_resource_server.items()
        }
        self._client = FlowsClient.new_client(
            client_id=GlobusWorkflowClient.CLIENT_ID,
            authorizer=self._authorizerMapping['flows.globus.org'],
            authorizer_callback=self._authorizerRetriever)

    def _authorizerRetriever(self, flow_url: str, flow_scope: str,
                             client_id: str) -> RefreshTokenAuthorizer:
        logger.debug(f'Searching for flow scope {flow_scope}')
        tokenData = self._tokenResponse.by_scopes[flow_scope]
        logger.debug(f'Found {tokenData}')
        resourceServer = str(tokenData['resource_server'])
        logger.debug(f'Resource Server is {resourceServer}')
        return self._authorizerMapping[resourceServer]

    def listFlowRuns(self) -> list[WorkflowRun]:
        runList: list[WorkflowRun] = list()
        flowID = str(self._settings.flowID.value)
        orderings = {'start_time': 'desc'}  # ordering by start_time (descending)
        response = self._client.list_flow_runs(flow_id=flowID, orderings=orderings)
        logger.debug(f'Flow Run List: {response}')

        # FIXME display_status -> current action
        # FIXME source/compute/results endpoints
        # FIXME 10 second polling
        for runDict in response['runs']:
            runID = runDict.get('run_id', '')
            runURL = f'https://app.globus.org/runs/{runID}/logs'
            run = WorkflowRun(label=runDict.get('label', ''),
                              startTime=runDict.get('start_time', ''),
                              completionTime=runDict.get('completion_time', ''),
                              status=runDict.get('status', ''),
                              action=runDict.get('display_status', ''),
                              runID=runID,
                              runURL=runURL)
            runList.append(run)

        #flow_action_id = response.data['actions'][0]['action_id']
        #flow_action = flows_client.flow_action_status(flow_id, flow_scope,
        #                                              flow_action_id)
        #try:
        #    flow_step = flow_action['details']['action_statuses'][0]['state_name']
        #except:
        #    flow_step = 'None'

        #flow_status = flow_action['status']
        #scanid = re.search(r'^\d+', flow_action['label'])[0]
        #info = f"{scanid} {flow_step} {flow_status}"

        return runList

    def _loadFlowDefinition(self) -> dict[str, Any]:
        flowDefinitionPath = Path(__file__).parents[0] / 'flowDefinition.json'

        with flowDefinitionPath.open(mode='r') as fp:
            flowDefinition = json.load(fp)

        return flowDefinition

    def _loadInputSchema(self) -> dict[str, Any]:
        inputSchemaPath = Path(__file__).parents[0] / 'inputSchema.json'

        with inputSchemaPath.open(mode='r') as fp:
            inputSchema = json.load(fp)

        return inputSchema

    def deployFlow(self) -> UUID:
        flowDefinition = self._loadFlowDefinition()
        inputSchema = self._loadInputSchema()

        response = self._client.deploy_flow(
            flow_definition=flowDefinition,
            title='Ptychodus',
            input_schema=inputSchema,
            visible_to=['public'],
            runnable_by=['all_authenticated_users'],
        )

        logger.debug(f'Deploy Flow Response: {response}')
        return UUID(response.data['id'])

    def updateFlow(self, flowID: UUID) -> None:
        flowDefinition = self._loadFlowDefinition()
        inputSchema = self._loadInputSchema()

        response = self._client.update_flow(
            flow_id=flowID,
            flow_definition=flowDefinition,
            title='Ptychodus',
            input_schema=inputSchema,
            visible_to=['public'],
            runnable_by=['all_authenticated_users'],
        )

        logger.debug(f'Update Flow Response: {response}')

    def listFlows(self) -> None:
        response = self._client.list_flows()
        logger.info(f'Flow List: {response}')

    def deleteFlow(self, flowID: UUID) -> None:
        response = self._client.delete_flow(flowID)
        logger.info(f'Delete Flow Response: {response}')

    def runFlow(self) -> None:
        flowID = str(self._settings.flowID.value)
        flowScope = None
        flowInput = {'input': {'echo_string': 'From Ptychodus', 'sleep_time': 10}}

        runResponse = self._client.run_flow(flowID, flowScope, flowInput, label='Run Label')
        logger.info(f'Run Flow Response: {json.dumps(runResponse.data, indent=4)}')

        # TODO enumerate_runs
        # TODO flow_action_cancel
        # TODO flow_action_log
        # TODO flow_action_release
        # TODO flow_action_resume
        # TODO flow_action_status
        # TODO flow_action_update
        # TODO update_runs


class GlobusWorkflowClientBuilder(WorkflowClientBuilder):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__(settings)
        self._authClient: Optional[NativeAppAuthClient] = None

    def getAuthorizeURL(self) -> str:
        if self._authClient is None:
            gacVersion = version('globus-automate-client')
            logger.info(f'\tGlobus Automate Client {gacVersion}')
            self._authClient = NativeAppAuthClient(GlobusWorkflowClient.CLIENT_ID)

        FLOW_ID = str(self._settings.flowID.value)
        FLOW_ID_ = FLOW_ID.replace('-', '_')

        requestedScopes = [
            MANAGE_FLOWS_SCOPE,
            RUN_FLOWS_SCOPE,
            RUN_STATUS_SCOPE,
            VIEW_FLOWS_SCOPE,
            f'https://auth.globus.org/scopes/{FLOW_ID}/flow_{FLOW_ID_}_user',
        ]

        logger.debug(f'Requested scopes: {requestedScopes}')

        self._authClient.oauth2_start_flow(requested_scopes=requestedScopes, refresh_tokens=True)
        return self._authClient.oauth2_get_authorize_url()

    def build(self, authCode: str) -> GlobusWorkflowClient:
        if self._authClient is None:
            raise RuntimeError('Missing AuthClient!')

        tokenResponse = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        logger.debug(f'Token response: {tokenResponse}')
        return GlobusWorkflowClient(self._settings, self._authClient, tokenResponse)
