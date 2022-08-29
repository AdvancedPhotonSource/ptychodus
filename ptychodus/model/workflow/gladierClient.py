from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, Optional
from uuid import UUID
import json
import logging
import pprint

from funcx import FuncXClient
from gladier import generate_flow_definition, GladierBaseClient
from globus_automate_client import FlowsClient
from globus_automate_client.flows_client import ALL_FLOW_SCOPES
from globus_sdk import NativeAppAuthClient, OAuthTokenResponse, RefreshTokenAuthorizer
from globus_sdk.scopes import AuthScopes, SearchScopes

from .client import WorkflowClient, WorkflowClientBuilder, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)

CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'


@generate_flow_definition
class GladierTestClient(GladierBaseClient):
    client_id = CLIENT_ID
    gladier_tools = [
        "gladier_tools.posix.shell_cmd.ShellCmdTool",
    ]


class GlobusWorkflowClient(WorkflowClient):

    def __init__(self, settings: WorkflowSettings, authorizerDict: dict[str, Any]) -> None:
        super().__init__()
        self._settings = settings
        self._client = GladierTestClient(authorizers=authorizerDict,
                                         auto_login=False,
                                         auto_registration=True)
        logger.debug(f'Client Scopes: {self._client.scopes}')

    def listFlowRuns(self) -> list[WorkflowRun]:
        runList: list[WorkflowRun] = list()
        flowID = str(self._settings.flowID.value)
        orderings = {'start_time': 'desc'}  # ordering by start_time (descending)
        response = self._client.flows_client.list_flow_runs(flow_id=flowID, orderings=orderings)
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

        response = self._client.flows_client.deploy_flow(
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

        response = self._client.flows_client.update_flow(
            flow_id=flowID,
            flow_definition=flowDefinition,
            title='Ptychodus',
            input_schema=inputSchema,
            visible_to=['public'],
            runnable_by=['all_authenticated_users'],
        )

        logger.debug(f'Update Flow Response: {response}')

    def listFlows(self) -> None:
        response = self._client.flows_client.list_flows()
        logger.info(f'Flow List: {response}')

    def deleteFlow(self, flowID: UUID) -> None:
        response = self._client.flows_client.delete_flow(flowID)
        logger.info(f'Delete Flow Response: {response}')

    def runFlow(self) -> None:
        flowID = str(self._settings.flowID.value)
        flowScope = None
        flowInput = {'input': {'echo_string': 'From Ptychodus', 'sleep_time': 10}}

        runResponse = self._client.run_flow(flowInput, label='Run Label')
        logger.info(f'Run Flow Response: {json.dumps(runResponse.data, indent=4)}')


class GlobusWorkflowClientBuilder(WorkflowClientBuilder):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__(settings)
        self._authClient: Optional[NativeAppAuthClient] = None

    def getAuthorizeURL(self) -> str:
        if self._authClient is None:
            self._authClient = NativeAppAuthClient(CLIENT_ID)

        FLOW_ID = str(self._settings.flowID.value)
        FLOW_ID_ = FLOW_ID.replace('-', '_')

        requestedScopes = [
            # FuncX Scopes
            FuncXClient.FUNCX_SCOPE,
            AuthScopes.openid,
            SearchScopes.all,

            # Automate scopes
            *ALL_FLOW_SCOPES,

            # Flow scope
            f'https://auth.globus.org/scopes/{FLOW_ID}/flow_{FLOW_ID_}_user',
        ]

        logger.debug('Requested Scopes: {pprint.pformat(requestedScopes)}')

        self._authClient.oauth2_start_flow(requested_scopes=requestedScopes, refresh_tokens=True)
        return self._authClient.oauth2_get_authorize_url()

    def build(self, authCode: str) -> GlobusWorkflowClient:
        if self._authClient is None:
            raise RuntimeError('Missing AuthClient!')

        tokenResponse = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        logger.debug(f'Token response: {tokenResponse}')
        authorizerDict: dict[str, Any] = dict()

        for resourceServer, tokenData in tokenResponse.by_resource_server.items():
            authorizer = RefreshTokenAuthorizer(
                refresh_token=tokenData['refresh_token'],
                auth_client=self._authClient,
                access_token=tokenData['access_token'],
                expires_at=tokenData['expires_at_seconds'],
            )

            for scope in tokenData['scope'].split():
                authorizerDict[scope] = authorizer

        logger.debug('Authorizers: {pprint.pformat(authorizerDict)}')

        return GlobusWorkflowClient(self._settings, authorizerDict)
