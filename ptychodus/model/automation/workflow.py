from pathlib import Path

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
        diffractionFilePath = next(filePath.parent.glob('*.tif'))
        diffractionFileFilter = 'Tagged Image File Format Files (*.tif *.tiff)'
        scanFilePath = filePath
        scanFileFilter = 'EPICS MDA Files (*.mda)'
        scanName = filePath.stem
        flowLabel = filePath.parent.name

        self._dataCore.loadDiffractionDataset(diffractionFilePath, diffractionFileFilter)
        self._scanCore.openScan(scanFilePath, scanFileFilter)
        self._scanCore.setActiveScan(scanName)
        self._probeCore.initializeAndActivateProbe()
        self._objectCore.initializeAndActivateObject()
        self._workflowCore.executeWorkflow(flowLabel)
