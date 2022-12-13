from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
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

from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .api import WorkflowAuthorizer, WorkflowExecutor, WorkflowRun, WorkflowThread
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
        logger.debug(f'Authenticate at {authorizeURL}')
        self._authorizeURL = authorizeURL
        self.isAuthorizedEvent.clear()
        # TODO figure out how to shut down nicely
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
        'gladier_tools.globus.transfer.Transfer:Settings',
        'gladier_tools.globus.transfer.Transfer:State',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:Results',
        # TODO 'gladier_tools.publish.Publish',
    ]


class CustomCodeHandler(fair_research_login.CodeHandler):

    def __init__(self, authorizer: GlobusWorkflowAuthorizer) -> None:
        self._authorizer = authorizer

    def authenticate(self, url: str) -> str:
        return self._authorizer.authenticate(url)

    def get_code(self) -> str:
        return self._authorizer.getCodeFromAuthorizeURL()


@dataclass(frozen=True)
class WorkflowInput:
    label: str
    flowInput: Mapping[str, Any]


class GlobusWorkflowThread(WorkflowThread, threading.Thread):

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
            token_storage=fair_research_login.JSONTokenStorage('mytokens.json'),
        )
        self._gladierClient: Optional[gladier.GladierBaseClient] = None
        self._inputQueue: queue.Queue[WorkflowInput] = queue.Queue()
        self._stopEvent = threading.Event()
        self.isAuthorizedEvent = threading.Event()

    def _requestAuthorization(self, scopes: list[str]) -> ScopeAuthorizerMapping:
        logger.debug('Requested authorization scopes: {pformat(scopes)}')

        # 'force' is used for any underlying scope changes. For example, if a flow adds transfer
        # functionality since it was last run, running it again would require a re-login.
        self._authClient.login(requested_scopes=scopes, force=True, refresh_tokens=True)
        return self._authClient.get_authorizers_by_scope()

    def _requireClient(self) -> None:
        if self._gladierClient is None:
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

            self._gladierClient = PtychodusClient(login_manager=loginManager)

    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        runList: list[WorkflowRun] = list()

        # FIXME listFlowRuns
        # response = self._gladierClient.list_flow_runs(
        #     flow_id=self._gladierClient.flows_manager.get_flow_id(),
        #     orderings={'start_time': 'desc'},  # order by start_time (descending)
        # )
        # logger.debug(f'Flow Run List: {response}')
        #
        # for runDict in response['runs']:
        #     action = runDict.get('display_status', '')  # TODO display_status -> current action
        #     runID = runDict.get('run_id', '')
        #     run = WorkflowRun(
        #         label=runDict.get('label', ''),
        #         startTime=runDict.get('start_time', ''),
        #         completionTime=runDict.get('completion_time', ''),
        #         status=runDict.get('status', ''),
        #         action=action,
        #         runID=runID,
        #         runURL=f'https://app.globus.org/runs/{runID}/logs',
        #     )
        #     runList.append(run)

        return runList

    def runFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        input_ = WorkflowInput(label, flowInput)
        self._inputQueue.put(input_)

    def _runFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        self._requireClient()

        if self._gladierClient is not None:
            response = self._gladierClient.run_flow(
                flow_input={'input': flowInput},
                label=label,
                tags=['aps', 'ptychography'],
            )
            logger.info(f'Run Flow Response: {json.dumps(response.data, indent=4)}')

    def run(self) -> None:
        while not self._stopEvent.is_set():
            try:
                # TODO custom timeout
                input_ = self._inputQueue.get(block=True, timeout=1)

                try:
                    self._runFlow(input_.label, input_.flowInput)
                finally:
                    self._inputQueue.task_done()
            except queue.Empty:
                pass
            except:
                logger.exception('Error running flow!')

    def start(self) -> None:
        threading.Thread.start(self)

    def stop(self) -> None:
        self._inputQueue.join()

        logger.info('Stopping workflow thread...')
        self._stopEvent.set()

        self.join()

        with self._inputQueue.mutex:
            self._inputQueue.queue.clear()

        logger.info('Workflow thread stopped.')


class GlobusWorkflowExecutor(WorkflowExecutor):

    def __init__(self, settings: WorkflowSettings, thread: WorkflowThread,
                 settingsRegistry: SettingsRegistry, stateDataRegistry: StateDataRegistry) -> None:
        super().__init__()
        self._settings = settings
        self._thread = thread
        self._settingsRegistry = settingsRegistry
        self._stateDataRegistry = stateDataRegistry

    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        return self._thread.listFlowRuns()

    def runFlow(self, label: str) -> None:
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
            'settings_transfer_source_endpoint_id':
            str(self._settings.inputDataEndpointID.value),
            'settings_transfer_source_path':
            f'{self._settings.inputDataGlobusPath.value}/{settingsFileName}',
            'settings_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'settings_transfer_destination_path':
            str(self._settings.computeDataGlobusPath.value),
            'settings_transfer_recursive':
            False,
            'state_transfer_source_endpoint_id':
            str(self._settings.inputDataEndpointID.value),
            'state_transfer_source_path':
            f'{self._settings.inputDataGlobusPath.value}/{restartFileName}',
            'state_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'state_transfer_destination_path':
            str(self._settings.computeDataGlobusPath.value),
            'state_transfer_recursive':
            False,
            'funcx_endpoint_compute':
            str(self._settings.computeFuncXEndpointID.value),
            'ptychodus_restart_file':
            f'{self._settings.computeDataPosixPath.value}/{restartFileName}',
            'ptychodus_settings_file':
            f'{self._settings.computeDataPosixPath.value}/{settingsFileName}',
            'results_transfer_source_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'results_transfer_source_path':
            f'{self._settings.computeDataGlobusPath.value}/{resultsFileName}',
            'results_transfer_destination_endpoint_id':
            str(self._settings.outputDataEndpointID.value),
            'results_transfer_destination_path':
            str(self._settings.outputDataGlobusPath.value),
            'results_transfer_recursive':
            False
        }

        self._thread.runFlow(label, flowInput)
