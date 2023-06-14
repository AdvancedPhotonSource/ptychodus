from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import logging
import queue

from ...api.settings import SettingsRegistry
from ...api.state import StateDataRegistry
from .locator import DataLocator
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowJob:
    flowLabel: str
    flowInput: Mapping[str, Any]


class WorkflowExecutor:

    def __init__(self, settings: WorkflowSettings, inputDataLocator: DataLocator,
                 computeDataLocator: DataLocator, outputDataLocator: DataLocator,
                 settingsRegistry: SettingsRegistry, stateDataRegistry: StateDataRegistry) -> None:
        super().__init__()
        self._settings = settings
        self._inputDataLocator = inputDataLocator
        self._computeDataLocator = computeDataLocator
        self._outputDataLocator = outputDataLocator
        self._settingsRegistry = settingsRegistry
        self._stateDataRegistry = stateDataRegistry
        self.jobQueue: queue.Queue[WorkflowJob] = queue.Queue()

    def runFlow(self, flowLabel: str) -> None:
        transferSyncLevel = 3  # Copy files if checksums of the source and destination do not match.

        inputDataPosixPath = self._inputDataLocator.getPosixPath() / flowLabel
        computeDataPosixPath = self._computeDataLocator.getPosixPath() / flowLabel

        inputDataGlobusPath = f'{self._inputDataLocator.getGlobusPath()}/{flowLabel}'
        computeDataGlobusPath = f'{self._computeDataLocator.getGlobusPath()}/{flowLabel}'
        outputDataGlobusPath = f'{self._outputDataLocator.getGlobusPath()}/{flowLabel}'

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
            str(self._inputDataLocator.getEndpointID()),
            'input_data_transfer_source_path':
            inputDataGlobusPath,
            'input_data_transfer_destination_endpoint_id':
            str(self._computeDataLocator.getEndpointID()),
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
            str(self._computeDataLocator.getEndpointID()),
            'output_data_transfer_source_path':
            f'{computeDataGlobusPath}/{resultsFileName}',
            'output_data_transfer_destination_endpoint_id':
            str(self._outputDataLocator.getEndpointID()),
            'output_data_transfer_destination_path':
            f'{outputDataGlobusPath}/{resultsFileName}',
            'output_data_transfer_recursive':
            False
        }

        input_ = WorkflowJob(flowLabel, flowInput)
        self.jobQueue.put(input_)
