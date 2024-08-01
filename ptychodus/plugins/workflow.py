from pathlib import Path
import csv
import logging
import re

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

logger = logging.getLogger(__name__)


class APS2IDFileBasedWorkflow(FileBasedWorkflow):

    @property
    def isWatchRecursive(self) -> bool:
        return False

    def getWatchFilePattern(self) -> str:
        return '*.csv'

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])

        diffractionFilePath = filePath.parents[1] / 'raw_data' / f'scan{scanID}_master.h5'
        workflowAPI.openPatterns(diffractionFilePath, fileType='NeXus')
        productAPI = workflowAPI.createProduct(f'scan{scanID}')
        productAPI.openScan(filePath, fileType='CSV')
        productAPI.buildProbe('Disk')
        productAPI.buildObject('Random')
        productAPI.reconstruct()


class APS26IDFileBasedWorkflow(FileBasedWorkflow):

    @property
    def isWatchRecursive(self) -> bool:
        return False

    def getWatchFilePattern(self) -> str:
        return '*.mda'

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        scanName = filePath.stem
        scanID = int(re.findall(r'\d+', scanName)[-1])

        diffractionDirPath = filePath.parents[1] / 'h5'

        for diffractionFilePath in diffractionDirPath.glob(f'scan_{scanID}_*.h5'):
            digits = int(re.findall(r'\d+', diffractionFilePath.stem)[-1])

            if digits != 0:
                break

        workflowAPI.openPatterns(diffractionFilePath, fileType='HDF5')
        productAPI = workflowAPI.createProduct(f'scan_{scanID}')
        productAPI.openScan(filePath, fileType='MDA')
        productAPI.buildProbe('Disk')
        productAPI.buildObject('Random')
        productAPI.reconstruct()


class APS31IDEFileBasedWorkflow(FileBasedWorkflow):

    @property
    def isWatchRecursive(self) -> bool:
        return True

    def getWatchFilePattern(self) -> str:
        return '*.h5'

    def execute(self, workflowAPI: WorkflowAPI, filePath: Path) -> None:
        experimentDir = filePath.parents[3]
        scanID = int(re.findall(r'\d+', filePath.stem)[0])
        scanFile = experimentDir / 'scan_positions' / f'scan_{scanID:05d}.dat'
        scanNumbersFile = experimentDir / 'dat-files' / 'tomography_scannumbers.txt'
        scanLabel = ''

        with scanNumbersFile.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=' ')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) != 7:
                    logger.warning('Unexpected row in tomography_scannumbers.txt!')
                    logger.debug(row)
                    continue

                try:
                    rowID = int(row[0])
                except ValueError:
                    logger.warning('Failed to parse row ID in tomography_scannumbers.txt!')
                    logger.debug(row[0])
                    continue

                if rowID == scanID:
                    scanLabel = str(row[6])
                    break

        if scanLabel:
            logger.debug(f'{filePath.stem} -> {scanID} -> {scanLabel}')

            workflowAPI.openPatterns(filePath, fileType='LYNX')
            productAPI = workflowAPI.createProduct(scanLabel)
            productAPI.openScan(scanFile, fileType='LYNXOrchestra')
            productAPI.buildProbe('fresnel_zone_plate')
            productAPI.buildObject('random')
            print(f'Reconstruct {scanLabel}!')  # FIXME productAPI.reconstruct()
        else:
            logger.warning(f'Failed to locate label for {rowID}!')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.fileBasedWorkflows.registerPlugin(
        APS2IDFileBasedWorkflow(),
        simpleName='APS2ID',
        displayName='APS 2-ID',
    )
    registry.fileBasedWorkflows.registerPlugin(
        APS26IDFileBasedWorkflow(),
        simpleName='APS26ID',
        displayName='APS 26-ID',
    )
    registry.fileBasedWorkflows.registerPlugin(
        APS31IDEFileBasedWorkflow(),
        simpleName='APS31IDE',
        displayName='APS 31-ID-E',
    )
