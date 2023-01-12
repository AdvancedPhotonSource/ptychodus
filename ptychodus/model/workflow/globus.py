from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from pprint import pformat, pprint
from typing import Any, Final, Optional, Union
import json
import logging
import queue
import threading

import fair_research_login
import gladier
import gladier.managers
import globus_sdk

from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .api import WorkflowAuthorizer, WorkflowExecutor, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)

AuthorizerTypes = Union[globus_sdk.AccessTokenAuthorizer, globus_sdk.RefreshTokenAuthorizer]
ScopeAuthorizerMapping = Mapping[str, AuthorizerTypes]


class GlobusWorkflowAuthorizer(WorkflowAuthorizer):
    CLIENT_ID: Final[str] = '5c0fb474-ae53-44c2-8c32-dd0db9965c57'

    def __init__(self) -> None:
        super().__init__()
        self._authorizeCode = str()
        self._authorizeURL = str()
        self.isAuthorizedEvent = threading.Event()
        self.isAuthorizedEvent.set()

    @property
    def isAuthorized(self) -> bool:
        return self.isAuthorizedEvent.is_set()

    def getAuthorizeURL(self) -> str:
        return self._authorizeURL

    def setCodeFromAuthorizeURL(self, authorizeCode: str) -> None:
        self._authorizeCode = authorizeCode
        self.isAuthorizedEvent.set()

    def getCodeFromAuthorizeURL(self) -> str:
        return self._authorizeCode

    def authenticate(self, authorizeURL: str) -> str:
        logger.info(f'Authenticate at {authorizeURL}')
        self._authorizeURL = authorizeURL
        self.isAuthorizedEvent.clear()
        # FIXME make sure that we can still shut down nicely
        self.isAuthorizedEvent.wait()
        return self._authorizeCode


def ptychodus_reconstruct(**data: str) -> None:
    from pathlib import Path
    from ptychodus.model import ModelArgs, ModelCore

    modelArgs = ModelArgs(
        restartFilePath=Path(data['ptychodus_restart_file']),
        settingsFilePath=Path(data['ptychodus_settings_file']),
    )

    with ModelCore(modelArgs) as model:
        model.batchModeReconstruct()


@gladier.generate_flow_definition
class PtychodusReconstruct(gladier.GladierBaseTool):
    funcx_functions = [ptychodus_reconstruct]
    required_input = [
        'ptychodus_restart_file',
        'ptychodus_settings_file',
    ]


@gladier.generate_flow_definition
class PtychodusClient(gladier.GladierBaseClient):
    client_id = GlobusWorkflowAuthorizer.CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:Input',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:Results',
        # TODO 'gladier_tools.publish.Publish',
    ]


class CustomCodeHandler(fair_research_login.CodeHandler):

    def __init__(self, authorizer: GlobusWorkflowAuthorizer) -> None:
        super().__init__()
        self._authorizer = authorizer
        self.set_browser_enabled(False)

    def authenticate(self, url: str) -> str:
        return self._authorizer.authenticate(url)

    def get_code(self) -> str:
        return self._authorizer.getCodeFromAuthorizeURL()


@dataclass(frozen=True)
class WorkflowInput:
    label: str
    flowInput: Mapping[str, Any]


class GlobusWorkflowThread(threading.Thread):

    def __init__(self, settings: WorkflowSettings, authorizer: GlobusWorkflowAuthorizer) -> None:
        super().__init__()
        logger.info('\tGlobus SDK ' + version('globus-sdk'))
        logger.info('\tFair Research Login ' + version('fair-research-login'))
        logger.info('\tGladier ' + version('gladier'))
        self._settings = settings
        self._authClient = fair_research_login.NativeClient(
            client_id=GlobusWorkflowAuthorizer.CLIENT_ID,
            app_name='Ptychodus',
            code_handlers=[CustomCodeHandler(authorizer)],
        )
        self.__gladierClient: Optional[gladier.GladierBaseClient] = None
        self._inputQueue: queue.Queue[WorkflowInput] = queue.Queue()
        self._stopEvent = threading.Event()
        self.isAuthorizedEvent = threading.Event()

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

    def _listFlowRuns(self) -> Sequence[Mapping[str, Any]]:
        flowsManager = self._gladierClient.flows_manager
        flowsClient = flowsManager.flows_client
        flowID = flowsManager.get_flow_id()
        response = flowsClient.list_flow_runs(flowID)
        runDictList = response['runs']

        while response['has_next_page']:
            response = flowsClient.list_flow_runs(flowID, marker=response['marker'])
            runDictList.extend(response['runs'])

        return runDictList

    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        runList: list[WorkflowRun] = list()
        runDictSequence = self._listFlowRuns()

        for runDict in runDictSequence:
            runID = runDict.get('run_id', '')
            action = self._getCurrentAction(runID)
            run = WorkflowRun(
                label=runDict.get('label', ''),
                startTime=runDict.get('start_time', ''),
                completionTime=runDict.get('completion_time', ''),
                status=runDict.get('status', ''),
                action=action,
                runID=runID,
                runURL=f'https://app.globus.org/runs/{runID}/logs',
            )
            runList.append(run)

        return runList

    def enqueueFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        input_ = WorkflowInput(label, flowInput)
        self._inputQueue.put(input_)

    def _runFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        response = self._gladierClient.run_flow(
            flow_input={'input': flowInput},
            label=label,
            tags=['aps', 'ptychography'],
        )
        logger.info(f'Run Flow Response: {json.dumps(response, indent=4)}')

    def run(self) -> None:
        while not self._stopEvent.is_set():
            try:
                # FIXME custom timeout
                input_ = self._inputQueue.get(block=True, timeout=1)

                try:
                    self._runFlow(input_.label, input_.flowInput)
                finally:
                    self._inputQueue.task_done()
            except queue.Empty:
                pass
            except:
                logger.exception('Error running flow!')

    def stop(self) -> None:
        self._inputQueue.join()

        logger.info('Stopping workflow thread...')
        self._stopEvent.set()

        self.join()

        with self._inputQueue.mutex:
            self._inputQueue.queue.clear()

        logger.info('Workflow thread stopped.')


class GlobusWorkflowExecutor(WorkflowExecutor):

    def __init__(self, settings: WorkflowSettings, authorizer: GlobusWorkflowAuthorizer,
                 settingsRegistry: SettingsRegistry, stateDataRegistry: StateDataRegistry) -> None:
        super().__init__()
        self._settings = settings
        self._authorizer = authorizer
        self._settingsRegistry = settingsRegistry
        self._stateDataRegistry = stateDataRegistry
        self._thread = GlobusWorkflowThread(settings, authorizer)

    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        return self._thread.listFlowRuns()

    def runFlow(self, label: str) -> None:
        transferSyncLevel = 3  # Copy files if checksums of the source and destination do not match.
        settingsFileName = 'input.ini'
        restartFileName = 'input.npz'
        resultsFileName = 'output.npz'
        inputDataPosixPath = Path(self._settings.inputDataPosixPath.value) / label

        try:
            inputDataPosixPath.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.error('Input data POSIX path must be a directory!')
            return

        self._settingsRegistry.saveSettings(inputDataPosixPath / settingsFileName)
        self._stateDataRegistry.saveStateData(inputDataPosixPath / restartFileName,
                                              restartable=True)

        flowInput = {
            'input_transfer_source_endpoint_id': str(self._settings.inputDataEndpointID.value),
            'input_transfer_source_path': f'{self._settings.inputDataGlobusPath.value}/{label}',
            'input_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'input_transfer_destination_path': str(self._settings.computeDataGlobusPath.value),
            'input_transfer_recursive': True,
            'input_transfer_sync_level': transferSyncLevel,
            'funcx_endpoint_compute': str(self._settings.computeFuncXEndpointID.value),
            'ptychodus_restart_file':
            f'{self._settings.computeDataPosixPath.value}/{label}/{restartFileName}',
            'ptychodus_settings_file':
            f'{self._settings.computeDataPosixPath.value}/{label}/{settingsFileName}',
            'results_transfer_source_endpoint_id': str(self._settings.computeDataEndpointID.value),
            'results_transfer_source_path':
            f'{self._settings.computeDataGlobusPath.value}/{label}/{resultsFileName}',
            'results_transfer_destination_endpoint_id':
            str(self._settings.outputDataEndpointID.value),
            'results_transfer_destination_path': str(self._settings.outputDataGlobusPath.value),
            'results_transfer_recursive': False
        }

        self._thread.enqueueFlow(label, flowInput)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._thread.stop()
