from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import logging
import queue

from ...api.action import Action
from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowJob:
    flowLabel: str
    flowInput: Mapping[str, Any]


class WorkflowExecutor:

    def __init__(self, settings: WorkflowSettings, settingsRegistry: SettingsRegistry,
                 stateDataRegistry: StateDataRegistry) -> None:
        super().__init__()
        self._settings = settings
        self._settingsRegistry = settingsRegistry
        self._stateDataRegistry = stateDataRegistry
        self.jobQueue: queue.Queue[WorkflowJob] = queue.Queue()

    def runFlow(self, flowLabel: str) -> None:
        transferSyncLevel = 3  # Copy files if checksums of the source and destination do not match.

        inputDataPosixPath = self._settings.inputDataPosixPath.value / flowLabel
        computeDataPosixPath = self._settings.computeDataPosixPath.value / flowLabel
        outputDataPosixPath = self._settings.outputDataPosixPath.value / flowLabel

        inputDataGlobusPath = f'{self._settings.inputDataGlobusPath.value}/{flowLabel}'
        computeDataGlobusPath = f'{self._settings.computeDataGlobusPath.value}/{flowLabel}'
        outputDataGlobusPath = f'{self._settings.outputDataGlobusPath.value}/{flowLabel}'

        settingsFileName = 'input.ini'
        restartFileName = 'input.npz'
        resultsFileName = 'output.npz'

        try:
            inputDataPosixPath.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.error('Input data POSIX path must be a directory!')
            return

        self._settingsRegistry.saveSettings(inputDataPosixPath / settingsFileName)
        self._stateDataRegistry.saveStateData(inputDataPosixPath / restartFileName,
                                              restartable=True)

        flowInput = {
            'input_data_transfer_source_endpoint_id':
            str(self._settings.inputDataEndpointID.value),
            'input_data_transfer_source_path':
            inputDataGlobusPath,
            'input_data_transfer_destination_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'input_data_transfer_destination_path':
            computeDataGlobusPath,
            'input_data_transfer_recursive':
            True,
            'input_data_transfer_sync_level':
            transferSyncLevel,
            'funcx_endpoint_compute':
            str(self._settings.computeFuncXEndpointID.value),
            'ptychodus_restart_file':
            str(computeDataPosixPath / restartFileName),
            'ptychodus_settings_file':
            str(computeDataPosixPath / settingsFileName),
            'ptychodus_results_file':
            str(computeDataPosixPath / resultsFileName),
            'output_data_transfer_source_endpoint_id':
            str(self._settings.computeDataEndpointID.value),
            'output_data_transfer_source_path':
            f'{computeDataGlobusPath}/{resultsFileName}',
            'output_data_transfer_destination_endpoint_id':
            str(self._settings.outputDataEndpointID.value),
            'output_data_transfer_destination_path':
            f'{outputDataGlobusPath}/{resultsFileName}',
            'output_data_transfer_recursive':
            False
        }

        input_ = WorkflowJob(flowLabel, flowInput)
        self.jobQueue.put(input_)


class ExecuteWorkflow(Action):

    def __init__(self, executor: WorkflowExecutor) -> None:
        self._executor = executor
        self._flowLabel: Optional[str] = None

    @property
    def name(self) -> str:
        return 'Execute Workflow'

    def setFlowLabel(self, flowLabel: str) -> None:
        self._flowLabel = flowLabel

    def __call__(self) -> None:
        if self._flowLabel is None:
            logger.error('Flow label is required!')
        else:
            self._executor.runFlow(self._flowLabel)
