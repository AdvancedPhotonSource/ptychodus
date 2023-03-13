from pathlib import Path
import logging
import re

from ..data import DataCore
from ..object import ObjectAPI
from ..probe import ProbeCore
from ..scan import ScanAPI
from ..workflow import WorkflowCore

logger = logging.getLogger(__name__)


class AutomationDatasetWorkflow:

    def __init__(self, dataCore: DataCore, scanAPI: ScanAPI, probeCore: ProbeCore,
                 objectAPI: ObjectAPI, workflowCore: WorkflowCore) -> None:
        self._dataCore = dataCore
        self._scanAPI = scanAPI
        self._probeCore = probeCore
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

        diffractionFileFilter = 'Hierarchical Data Format 5 Files (*.h5 *.hdf5)'
        scanFilePath = filePath
        scanFileFilter = 'EPICS MDA Files (*.mda)'

        self._dataCore.loadDiffractionDataset(diffractionFilePath, diffractionFileFilter)

        scanItemNameList = self._scanAPI.insertScanIntoRepositoryFromFile(
            scanFilePath, scanFileFilter)

        if len(scanItemNameList) == 1:
            self._scanAPI.setActiveScan(scanItemNameList[0])
        else:
            logger.error(f'Scan file contains {len(scanItemNameList)} items!')
            return

        self._probeCore.initializeAndActivateProbe()
        self._objectAPI.initializeAndActivateObject('Random')
        self._workflowCore.executeWorkflow(flowLabel)
