from abc import ABC, abstractmethod
from pathlib import Path
import logging
import re

from ..patterns import PatternsAPI
from ..workflow import WorkflowCore

logger = logging.getLogger(__name__)


# TODO add parameter optimization workflow
class AutomationDatasetWorkflow(ABC):

    @abstractmethod
    def execute(self, filePath: Path) -> None:
        '''executes the workflow'''
        pass


class S26AutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, patternsAPI: PatternsAPI, workflowCore: WorkflowCore) -> None:
        self._patternsAPI = patternsAPI
        self._workflowCore = workflowCore

    def execute(self, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])
        # FIXME flowLabel = f'scan_{scanID}'

        diffractionDirPath = filePath.parents[1] / 'h5'

        for diffractionFilePath in diffractionDirPath.glob(f'scan_{scanID}_*.h5'):
            digits = int(re.findall(r'\d+', diffractionFilePath.stem)[-1])

            if digits != 0:
                break

        self._patternsAPI.openPatterns(diffractionFilePath, fileType='HDF5')
        # FIXME self._scanAPI.insertItemIntoRepositoryFromFile(filePath, fileType='MDA', selectItem=True)
        # NOTE reuse probe
        # FIXME self._objectAPI.selectNewItemFromInitializerSimpleName('Random')
        # FIXME self._workflowCore.executeWorkflow(inputProductIndex)


class S2AutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, patternsAPI: PatternsAPI, workflowCore: WorkflowCore) -> None:
        self._patternsAPI = patternsAPI
        self._workflowCore = workflowCore

    def execute(self, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])
        # FIXME flowLabel = f'scan{scanID}'

        diffractionFilePath = filePath.parents[1] / 'raw_data' / f'scan{scanID}_master.h5'
        self._patternsAPI.openPatterns(diffractionFilePath, fileType='NeXus')
        # FIXME self._scanAPI.insertItemIntoRepositoryFromFile(filePath, fileType='CSV', selectItem=True)
        # NOTE reuse probe
        # FIXME self._objectAPI.selectNewItemFromInitializerSimpleName('Random')
        # FIXME self._workflowCore.executeWorkflow(inputProductIndex)


class PtychoNNTrainingAutomationDatasetWorkflow(AutomationDatasetWorkflow):

    def __init__(self, patternsAPI: PatternsAPI) -> None:
        self._patternsAPI = patternsAPI

    def execute(self, filePath: Path) -> None:
        # TODO watch for ptychodus NPZ files
        self._patternsAPI.openPatterns(filePath, fileType='NPZ')
        # FIXME self._reconstructor.ingestTrainingData()
        # FIXME self._reconstructor.train()
