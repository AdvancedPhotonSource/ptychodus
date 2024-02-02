from abc import ABC, abstractmethod
from pathlib import Path
import logging
import re

from ...api.state import StateDataRegistry
from ..patterns import DiffractionDataAPI
from ..workflow import WorkflowCore

logger = logging.getLogger(__name__)


# TODO add parameter optimization workflow
class AutomationDatasetWorkflow(ABC):

    @abstractmethod
    def execute(self, filePath: Path) -> None:
        '''executes the workflow'''
        pass


class S26AutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, dataAPI: DiffractionDataAPI, workflowCore: WorkflowCore) -> None:
        self._dataAPI = dataAPI
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

        self._dataAPI.loadDiffractionDataset(diffractionFilePath, fileType='HDF5')
        self._scanAPI.insertItemIntoRepositoryFromFile(filePath, fileType='MDA', selectItem=True)
        # NOTE reuse probe
        self._objectAPI.selectNewItemFromInitializerSimpleName('Random')
        self._workflowCore.executeWorkflow(flowLabel)


class S2AutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, dataAPI: DiffractionDataAPI, workflowCore: WorkflowCore) -> None:
        self._dataAPI = dataAPI
        self._workflowCore = workflowCore

    def execute(self, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])
        flowLabel = f'scan{scanID}'

        diffractionFilePath = filePath.parents[1] / 'raw_data' / f'scan{scanID}_master.h5'
        self._dataAPI.loadDiffractionDataset(diffractionFilePath, fileType='NeXus')
        self._scanAPI.insertItemIntoRepositoryFromFile(filePath, fileType='CSV', selectItem=True)
        # NOTE reuse probe
        self._objectAPI.selectNewItemFromInitializerSimpleName('Random')
        self._workflowCore.executeWorkflow(flowLabel)


class PtychoNNTrainingAutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, registry: StateDataRegistry) -> None:
        self._registry = registry

    def execute(self, filePath: Path) -> None:
        # TODO watch for ptychodus NPZ files
        self._registry.openStateData(filePath)
        self._reconstructor.ingestTrainingData()
        self._reconstructor.train()
