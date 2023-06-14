from abc import ABC, abstractmethod
from pathlib import Path
import logging
import re

from ...api.state import StateDataRegistry
from ..data import DiffractionDataAPI
from ..object import ObjectAPI
from ..probe import ProbeAPI
from ..ptychonn import PtychoNNTrainingDatasetBuilder
from ..scan import ScanAPI
from ..workflow import WorkflowCore

logger = logging.getLogger(__name__)


# FIXME add parameter optimization workflow
class AutomationDatasetWorkflow(ABC):

    @abstractmethod
    def execute(self, filePath: Path) -> None:
        '''executes the workflow'''
        pass


class S26AutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, dataAPI: DiffractionDataAPI, scanAPI: ScanAPI, probeAPI: ProbeAPI,
                 objectAPI: ObjectAPI, workflowCore: WorkflowCore) -> None:
        self._dataAPI = dataAPI
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._workflowCore = workflowCore

    def execute(self, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])
        flowLabel = f'scan_{scanID}'

        diffractionDirPath = filePath.parents[1] / 'h5'

        for diffractionFilePath in diffractionDirPath.glob(f'scan_{scanID}_*.h5'):
            digits = int(re.findall(r'\d+', diffractionFilePath.stem)[-1])

            if digits != 0:
                break

        self._dataAPI.loadDiffractionDataset(diffractionFilePath, simpleFileType='HDF5')
        scanItemName = self._scanAPI.insertItemIntoRepositoryFromFile(filePath,
                                                                      simpleFileType='MDA')

        if scanItemName is not None:
            self._scanAPI.selectItem(scanItemName)

        # NOTE reuse probe
        self._objectAPI.selectNewItemFromInitializerSimpleName('Random')
        self._workflowCore.executeWorkflow(flowLabel)


class S02AutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, dataAPI: DiffractionDataAPI, scanAPI: ScanAPI, probeAPI: ProbeAPI,
                 objectAPI: ObjectAPI, workflowCore: WorkflowCore) -> None:
        self._dataAPI = dataAPI
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._workflowCore = workflowCore

    def execute(self, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])
        flowLabel = f'scan{scanID}'

        diffractionFilePath = filePath.parents[1] / 'raw_data' / f'scan{scanID}_master.h5'
        self._dataAPI.loadDiffractionDataset(diffractionFilePath, simpleFileType='NeXus')
        scanItemName = self._scanAPI.insertItemIntoRepositoryFromFile(filePath,
                                                                      simpleFileType='CSV')

        if scanItemName is not None:
            self._scanAPI.selectItem(scanItemName)

        # NOTE reuse probe
        self._objectAPI.selectNewItemFromInitializerSimpleName('Random')
        self._workflowCore.executeWorkflow(flowLabel)


class PtychoNNTrainingAutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, registry: StateDataRegistry,
                 builder: PtychoNNTrainingDatasetBuilder) -> None:
        self._registry = registry
        self._builder = builder

    def execute(self, filePath: Path) -> None:
        # FIXME watch for ptychodus NPZ files
        self._registry.openStateData(filePath)
        self._builder.incorporateTrainingDataFromActiveItems()
        # FIXME save or train
