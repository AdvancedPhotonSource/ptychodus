from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import logging
import queue

from ptychodus.api.settings import SettingsRegistry

from ..patterns import PatternsAPI
from ..product import ProductAPI
from .locator import DataLocator
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkflowJob:
    flowLabel: str
    flowInput: Mapping[str, Any]


class WorkflowExecutor:

    def __init__(
        self,
        settings: WorkflowSettings,
        inputDataLocator: DataLocator,
        computeDataLocator: DataLocator,
        outputDataLocator: DataLocator,
        settingsRegistry: SettingsRegistry,
        patternsAPI: PatternsAPI,
        productAPI: ProductAPI,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._inputDataLocator = inputDataLocator
        self._computeDataLocator = computeDataLocator
        self._outputDataLocator = outputDataLocator
        self._productAPI = productAPI
        self._settingsRegistry = settingsRegistry
        self._patternsAPI = patternsAPI
        self.jobQueue: queue.Queue[WorkflowJob] = queue.Queue()

    def runFlow(self, inputProductIndex: int) -> None:
        transferSyncLevel = 3  # Copy files if checksums of the source and destination mismatch
        ptychodusAction = "reconstruct"  # TODO or 'train'

        try:
            flowLabel = self._productAPI.getItemName(inputProductIndex)
        except IndexError:
            logger.warning(f"Failed access product for flow ({inputProductIndex=})!")
            return

        inputDataPosixPath = self._inputDataLocator.getPosixPath() / flowLabel
        computeDataPosixPath = self._computeDataLocator.getPosixPath() / flowLabel

        inputDataGlobusPath = f"{self._inputDataLocator.getGlobusPath()}/{flowLabel}"
        computeDataGlobusPath = f"{self._computeDataLocator.getGlobusPath()}/{flowLabel}"
        outputDataGlobusPath = f"{self._outputDataLocator.getGlobusPath()}/{flowLabel}"

        settingsFile = "settings.ini"
        patternsFile = "patterns.npz"
        inputFile = "product-in.npz"
        outputFile = "product-out.npz"

        try:
            inputDataPosixPath.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            logger.warning("Input data POSIX path must be a directory!")
            return

        # TODO use workflow API
        self._settingsRegistry.saveSettings(inputDataPosixPath / settingsFile)
        self._patternsAPI.exportProcessedPatterns(inputDataPosixPath / patternsFile)
        self._productAPI.saveProduct(inputProductIndex,
                                     inputDataPosixPath / inputFile,
                                     fileType="NPZ")

        flowInput = {
            "input_data_transfer_source_endpoint_id":
            str(self._inputDataLocator.getEndpointID()),
            "input_data_transfer_source_path":
            inputDataGlobusPath,
            "input_data_transfer_destination_endpoint_id":
            str(self._computeDataLocator.getEndpointID()),
            "input_data_transfer_destination_path":
            computeDataGlobusPath,
            "input_data_transfer_recursive":
            True,
            "input_data_transfer_sync_level":
            transferSyncLevel,
            "compute_endpoint":
            str(self._settings.computeEndpointID.getValue()),
            "ptychodus_action":
            ptychodusAction,
            "ptychodus_settings_file":
            str(computeDataPosixPath / settingsFile),
            "ptychodus_patterns_file":
            str(computeDataPosixPath / patternsFile),
            "ptychodus_input_file":
            str(computeDataPosixPath / inputFile),
            "ptychodus_output_file":
            str(computeDataPosixPath / outputFile),
            "output_data_transfer_source_endpoint_id":
            str(self._computeDataLocator.getEndpointID()),
            "output_data_transfer_source_path":
            f"{computeDataGlobusPath}/{outputFile}",
            "output_data_transfer_destination_endpoint_id":
            str(self._outputDataLocator.getEndpointID()),
            "output_data_transfer_destination_path":
            f"{outputDataGlobusPath}/{outputFile}",
            "output_data_transfer_recursive":
            False,
        }

        input_ = WorkflowJob(flowLabel, flowInput)
        self.jobQueue.put(input_)
