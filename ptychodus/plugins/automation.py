from pathlib import Path
import re

from ptychodus.api.automation import FileBasedWorkflow, WorkflowAPI
from ptychodus.api.plugins import PluginRegistry


class APS2IDFileBasedWorkflow(FileBasedWorkflow):

    def getFilePattern(self) -> str:
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

    def getFilePattern(self) -> str:
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


def registerPlugins(registry: PluginRegistry) -> None:
    registry.fileBasedWorkflows.registerPlugin(
        APS2IDFileBasedWorkflow(),
        simpleName='APS2ID',
        displayName='LYNX Catalyst Particle',
    )
    registry.fileBasedWorkflows.registerPlugin(
        APS26IDFileBasedWorkflow(),
        simpleName='APS26ID',
        displayName='CNM/APS Hard X-Ray Nanoprobe',
    )
