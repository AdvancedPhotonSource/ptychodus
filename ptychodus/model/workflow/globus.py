from collections.abc import Mapping
from importlib.metadata import version
from pprint import pformat
from typing import Any, Final, Optional, Union
import json
import logging
import queue
import threading

import fair_research_login
import gladier
import gladier.managers
import globus_sdk

from .authorizer import WorkflowAuthorizer
from .executor import WorkflowExecutor
from .status import WorkflowStatus, WorkflowStatusRepository

logger = logging.getLogger(__name__)

AuthorizerTypes = Union[globus_sdk.AccessTokenAuthorizer, globus_sdk.RefreshTokenAuthorizer]
ScopeAuthorizerMapping = Mapping[str, AuthorizerTypes]

CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'


def ptychodus_reconstruct(**data: str) -> None:
    from pathlib import Path
    from ptychodus.model import ModelArgs, ModelCore

    modelArgs = ModelArgs(
        restartFilePath=Path(data['ptychodus_restart_file']),
        settingsFilePath=Path(data['ptychodus_settings_file']),
    )

    resultsFilePath = Path(data['ptychodus_results_file'])

    with ModelCore(modelArgs) as model:
        model.batchModeReconstruct(resultsFilePath)


@gladier.generate_flow_definition
class PtychodusReconstruct(gladier.GladierBaseTool):
    funcx_functions = [ptychodus_reconstruct]
    required_input = [
        'ptychodus_restart_file',
        'ptychodus_settings_file',
        'ptychodus_results_file',
    ]


@gladier.generate_flow_definition
class PtychodusClient(gladier.GladierBaseClient):
    client_id = CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:InputData',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:OutputData',
        # TODO 'gladier_tools.publish.Publish',
    ]


class CustomCodeHandler(fair_research_login.CodeHandler):

    def __init__(self, authorizer: WorkflowAuthorizer) -> None:
        super().__init__()
        self._authorizer = authorizer
        self.set_browser_enabled(False)

    def authenticate(self, url: str) -> str:
        self._authorizer.authenticate(url)
        return self.get_code()

    def get_code(self) -> str:
        return self._authorizer.getCodeFromAuthorizeURL()


class GlobusWorkflowThread(threading.Thread):

    def __init__(self, authorizer: WorkflowAuthorizer, statusRepository: WorkflowStatusRepository,
                 executor: WorkflowExecutor) -> None:
        super().__init__()
        self._authorizer = authorizer
        self._statusRepository = statusRepository
        self._executor = executor

        logger.info('\tGlobus SDK ' + version('globus-sdk'))
        logger.info('\tFair Research Login ' + version('fair-research-login'))
        logger.info('\tGladier ' + version('gladier'))

        self._authClient = fair_research_login.NativeClient(
            client_id=CLIENT_ID,
            app_name='Ptychodus',
            code_handlers=[CustomCodeHandler(authorizer)],
        )
        self.__gladierClient: Optional[gladier.GladierBaseClient] = None

    def _requestAuthorization(self, scopes: list[str]) -> ScopeAuthorizerMapping:
        logger.debug(f'Requested authorization scopes: {pformat(scopes)}')

        # 'force' is used for any underlying scope changes. For example, if a flow adds transfer
        # functionality since it was last run, running it again would require a re-login.
        self._authClient.login(requested_scopes=scopes, force=True, refresh_tokens=True)
        return self._authClient.get_authorizers_by_scope()

    @property
    def _gladierClient(self) -> gladier.GladierBaseClient:
        if self.__gladierClient is None:
            try:
                # Try to use a previous login to avoid a new login flow
                initial_authorizers = self._authClient.get_authorizers_by_scope()
            except fair_research_login.LoadError:
                # Passing in an empty dict will trigger the callback below
                initial_authorizers = dict()

            loginManager = gladier.managers.CallbackLoginManager(
                authorizers=initial_authorizers,
                # If additional logins are needed, the callback is used.
                callback=self._requestAuthorization,
            )

            self.__gladierClient = PtychodusClient(login_manager=loginManager)

        return self.__gladierClient

    def _getCurrentAction(self, runID: str) -> str:
        status = self._gladierClient.get_status(runID)
        action = status.get('state_name')

        if not action:
            try:
                det = status['details']
            except:
                logger.exception('Unexpected flow status!')
                logger.error(pformat(status))
            else:
                if det.get('details') and det['details'].get('state_name'):
                    action = det['details']['state_name']
                elif det.get('details') and det['details'].get('output'):
                    action = list(det['details']['output'].keys())[0]
                elif det.get('action_statuses'):
                    action = det['action_statuses'][0]['state_name']
                elif det.get('code') == 'FlowStarting':
                    pass

        return action

    def _refreshStatus(self) -> None:
        statusList: list[WorkflowStatus] = list()
        flowsManager = self._gladierClient.flows_manager
        flowID = flowsManager.get_flow_id()
        flowsClient = flowsManager.flows_client
        response = flowsClient.list_flow_runs(flowID)
        runDictList = response['runs']

        while response['has_next_page']:
            response = flowsClient.list_flow_runs(flowID, marker=response['marker'])
            runDictList.extend(response['runs'])

        for runDict in runDictList:
            runID = runDict.get('run_id', '')
            action = self._getCurrentAction(runID)

            run = WorkflowStatus(
                label=runDict.get('label', ''),
                startTime=runDict.get('start_time', ''),
                completionTime=runDict.get('completion_time', ''),
                status=runDict.get('status', ''),
                action=action,
                runID=runID,
                runURL=f'https://app.globus.org/runs/{runID}/logs',
            )

            statusList.append(run)

        self._statusRepository.update(statusList)

    def run(self) -> None:
        while not self._authorizer.shutdownEvent.is_set():
            if self._statusRepository.refreshStatusEvent.is_set():
                self._refreshStatus()
                self._statusRepository.refreshStatusEvent.clear()

            try:
                input_ = self._executor.jobQueue.get(block=True, timeout=1)
            except queue.Empty:
                continue

            try:
                response = self._gladierClient.run_flow(
                    flow_input={'input': input_.flowInput},
                    label=input_.flowLabel,
                    tags=['aps', 'ptychography'],
                )
            except:
                logger.exception('Error running flow!')
            finally:
                logger.info(f'Run Flow Response: {json.dumps(response, indent=4)}')
                self._executor.jobQueue.task_done()
