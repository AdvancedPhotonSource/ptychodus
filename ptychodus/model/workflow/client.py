from importlib.metadata import version
from pathlib import Path
import json
import logging

from globus_automate_client import FlowsClient
from globus_automate_client.flows_client import FLOWS_CLIENT_ID
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

    def _authorizerRetriever(self, flow_url: str, flow_scope: str,
                             client_id: str) -> RefreshTokenAuthorizer:
        return self._authorizer

    def printFlows(self) -> None:
        myFlows = self._client.list_flows()
        # type(myFlows) is globus_sdk.response.GlobusHTTPResponse
        logger.info(f'Flow List: {myFlows}')

    def _loadFlowDefinition(self):  # TODO type hint _
        flowDefinitionPath = Path(__file__).parents[0] / 'flowDefinition.json'

        with flowDefinitionPath.open(mode='r') as fp:
            flowDefinition = json.load(fp)

        print(type(flowDefinition))

        return flowDefinition

    def _loadInputSchema(self):  # TODO type hint _
        inputSchemaPath = Path(__file__).parents[0] / 'inputSchema.json'

        with inputSchemaPath.open(mode='r') as fp:
            inputSchema = json.load(fp)

        print(type(inputSchema))

        return inputSchema

    def deployFlow(self) -> None:
        flowDefinition = self._loadFlowDefinition()
        inputSchema = self._loadInputSchema()

        deployResult = self._client.deploy_flow(
            flow_definition=flowDefinition,
            title='Deployed via Ptychodus',
            input_schema=inputSchema,
            visible_to=['public'],
            runnable_by=['all_authenticated_users'],
        )

        print(type(deployResult))
        flowID = deployResult.data['id']
        print(type(flowID))

        logger.info(f'Flow ID: {flowID}')
        self._settings.flowID.value = flowID

    def runFlow(self) -> None:
        flowID = str(self._settings.flowID.value)
        flowScope = None
        flowInput = {'echo_string': 'From Ptychodus', 'sleep_time': 10}

        runResult = self._client.run_flow(flowID, flowScope, flowInput)
        logger.info(f'Flow Run Result: {json.dumps(runResult.data, indent=4)}')


class WorkflowClientBuilder:

    def __init__(self, settings: WorkflowSettings) -> None:
        gacVersion = version('globus-automate-client')
        logger.debug(f'\tGlobus Automate Client {gacVersion}')

        self._settings = settings
        self._authClient = NativeAppAuthClient(WorkflowClient.CLIENT_ID)

    def getAuthorizeURL(self) -> str:
        CLIENT_ID = FLOWS_CLIENT_ID
        # FIXME CLIENT_ID = WorkflowClient.CLIENT_ID

        FLOW_ID = str(self._settings.flowID.value)
        FLOW_ID_ = FLOW_ID.replace('-', '_')

        requestedScopes = [
            f'https://auth.globus.org/scopes/{CLIENT_ID}/view_flows',
            f'https://auth.globus.org/scopes/{CLIENT_ID}/manage_flows',
            f'https://auth.globus.org/scopes/{CLIENT_ID}/run',
            # TODO f'https://auth.globus.org/scopes/{FLOW_ID}/flow_{FLOW_ID_}_user',
            'https://auth.globus.org/scopes/8465696e-6bd6-4c9b-b5f2-7d4a6b19c5c1/flow_8465696e_6bd6_4c9b_b5f2_7d4a6b19c5c1_user',
        ]

        logger.debug(f'Requested Scopes: {requestedScopes}')

        self._authClient.oauth2_start_flow(requested_scopes=requestedScopes, refresh_tokens=True)
        return self._authClient.oauth2_get_authorize_url()

    def build(self, authCode: str) -> WorkflowClient:
        tokenResponse = self._authClient.oauth2_exchange_code_for_tokens(authCode.strip())
        logger.debug(tokenResponse)
        flowsTokens = tokenResponse.by_resource_server['flows.globus.org']
        authorizer = RefreshTokenAuthorizer(flowsTokens['refresh_token'], self._authClient)
        return WorkflowClient(self._settings, authorizer)
