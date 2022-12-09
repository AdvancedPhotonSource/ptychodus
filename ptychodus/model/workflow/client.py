from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional
import json
import logging
import queue
import threading

from gladier.managers import CallbackLoginManager
import gladier

from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .api import WorkflowClient, WorkflowRun
from .authorizerRepository import GlobusAuthorizerRepository, ScopeAuthorizerMapping
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


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
    client_id = GlobusAuthorizerRepository.CLIENT_ID
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:Settings',
        'gladier_tools.globus.transfer.Transfer:State',
        PtychodusReconstruct,
        'gladier_tools.globus.transfer.Transfer:Results',
        # TODO 'gladier_tools.publish.Publish',
    ]


@dataclass(frozen=True)
class WorkflowInput:
    label: str
    flowInput: Mapping[str, Any]


class GlobusWorkflowThread(threading.Thread):

    def __init__(self, settings: WorkflowSettings,
                 authorizerRepository: GlobusAuthorizerRepository) -> None:
        super().__init__()
        logger.info('\tGladier ' + version('gladier'))
        self._settings = settings
        self._authorizerRepository = authorizerRepository
        self._client: Optional[gladier.GladierBaseClient] = None
        self._inputQueue: queue.Queue[WorkflowInput] = queue.Queue()
        self._stopEvent = threading.Event()

    def _authorize(self, scopes: list[str]) -> ScopeAuthorizerMapping:
        self._authorizerRepository.requestAuthorization(scopes)
        self._authorizerRepository.isAuthorizedEvent.wait()
        return self._authorizerRepository.getAuthorizers()

    def _requireClient(self) -> None:
        if self._client is None:
            #FLOW_ID = str(self._settings.flowID.value)
            #FLOW_ID_ = FLOW_ID.replace('-', '_')

            scopes = [
                gladier.FlowsManager.AVAILABLE_SCOPES,
                # TODO f'https://auth.globus.org/scopes/{FLOW_ID}/flow_{FLOW_ID_}_user',
            ]
            loginManager = CallbackLoginManager(
                initial_authorizers=self._authorize(scopes),
                callback=self._authorize,
            )
            self._client = PtychodusClient(login_manager=loginManager)

    def listFlowRuns(self) -> list[WorkflowRun]:
        runList: list[WorkflowRun] = list()
        self._requireClient()

        if self._client is not None:
            response = self._client.list_flow_runs(
                flow_id=self._client.flows_manager.get_flow_id(),
                orderings={'start_time': 'desc'},  # order by start_time (descending)
            )
            logger.debug(f'Flow Run List: {response}')

            for runDict in response['runs']:
                action = runDict.get('display_status', '')  # TODO display_status -> current action
                runID = runDict.get('run_id', '')
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

    def runFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        input_ = WorkflowInput(label, flowInput)
        self._inputQueue.put(input_)

    def _runFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        self._requireClient()

        if self._client is not None:
            response = self._client.run_flow(
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

    def stop(self) -> None:
        self._inputQueue.join()

        logger.info('Stopping workflow thread...')
        self._stopEvent.set()

        self.join()

        with self._inputQueue.mutex:
            self._inputQueue.queue.clear()

        logger.info('Workflow thread stopped.')


class GlobusClient(WorkflowClient):

    def __init__(self, settings: WorkflowSettings, settingsRegistry: SettingsRegistry,
                 stateDataRegistry: StateDataRegistry,
                 authorizerRepository: GlobusAuthorizerRepository) -> None:
        super().__init__()
        self._settings = settings
        self._settingsRegistry = settingsRegistry
        self._stateDataRegistry = stateDataRegistry
        self._thread = GlobusWorkflowThread(settings, authorizerRepository)

    def listFlowRuns(self) -> list[WorkflowRun]:
        flowRuns: list[WorkflowRun] = list()
        # FIXME GlobusClient::listFlowRuns
        return flowRuns

    def runFlow(self, label: str) -> None:
        settingsFileName = f'{label}.ini'
        restartFileName = f'{label}.in.npz'
        resultsFileName = f'{label}.out.npz'
        inputDataPosixPath = Path(self._settings.inputDataPosixPath.value)

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
