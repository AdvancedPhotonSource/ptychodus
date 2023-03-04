from pathlib import Path
import re

from ..data import DataCore
from ..object import ObjectCore
from ..probe import ProbeCore
from ..scan import ScanCore
from ..workflow import WorkflowCore


class AutomationDatasetWorkflow:

    def __init__(self, dataCore: DataCore, scanCore: ScanCore, probeCore: ProbeCore,
                 objectCore: ObjectCore, workflowCore: WorkflowCore) -> None:
        self._dataCore = dataCore
        self._scanCore = scanCore
        self._probeCore = probeCore
        self._objectCore = objectCore
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
        self._scanCore.openScan(scanFilePath, scanFileFilter)
        self._scanCore.setActiveScan(scanName)
        self._probeCore.initializeAndActivateProbe()
        self._objectCore.initializeAndActivateObject()
        self._workflowCore.executeWorkflow(flowLabel)
