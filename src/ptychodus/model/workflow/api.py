from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import CropCenter
from ptychodus.api.settings import PathPrefixChange, SettingsRegistry
from ptychodus.api.workflow import WorkflowAPI, WorkflowProductAPI

from ..patterns import PatternsAPI
from ..product import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from ..reconstructor import ReconstructorAPI
from .executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class ConcreteWorkflowProductAPI(WorkflowProductAPI):
    def __init__(
        self,
        productAPI: ProductAPI,
        scanAPI: ScanAPI,
        probeAPI: ProbeAPI,
        objectAPI: ObjectAPI,
        reconstructorAPI: ReconstructorAPI,
        executor: WorkflowExecutor,
        productIndex: int,
    ) -> None:
        self._productAPI = productAPI
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._reconstructorAPI = reconstructorAPI
        self._executor = executor
        self._productIndex = productIndex

    def openScan(self, filePath: Path, *, fileType: str | None = None) -> None:
        self._scanAPI.openScan(self._productIndex, filePath, fileType=fileType)

    def buildScan(
        self, builderName: str | None = None, builderParameters: Mapping[str, Any] = {}
    ) -> None:
        if builderName is None:
            self._scanAPI.buildScanFromSettings(self._productIndex)
        else:
            self._scanAPI.buildScan(self._productIndex, builderName, builderParameters)

    def openProbe(self, filePath: Path, *, fileType: str | None = None) -> None:
        self._probeAPI.openProbe(self._productIndex, filePath, fileType=fileType)

    def buildProbe(
        self, builderName: str | None = None, builderParameters: Mapping[str, Any] = {}
    ) -> None:
        if builderName is None:
            self._probeAPI.buildProbeFromSettings(self._productIndex)
        else:
            self._probeAPI.buildProbe(self._productIndex, builderName, builderParameters)

    def openObject(self, filePath: Path, *, fileType: str | None = None) -> None:
        self._objectAPI.openObject(self._productIndex, filePath, fileType=fileType)

    def buildObject(
        self, builderName: str | None = None, builderParameters: Mapping[str, Any] = {}
    ) -> None:
        if builderName is None:
            self._objectAPI.buildObjectFromSettings(self._productIndex)
        else:
            self._objectAPI.buildObject(self._productIndex, builderName, builderParameters)

    def reconstructLocal(self) -> WorkflowProductAPI:
        logger.debug(f'Reconstruct: index={self._productIndex}')
        outputProductIndex = self._reconstructorAPI.reconstruct(self._productIndex)

        return ConcreteWorkflowProductAPI(
            self._productAPI,
            self._scanAPI,
            self._probeAPI,
            self._objectAPI,
            self._reconstructorAPI,
            self._executor,
            outputProductIndex,
        )

    def reconstructRemote(self) -> None:
        logger.debug(f'Execute Workflow: index={self._productIndex}')
        self._executor.runFlow(self._productIndex)

    def saveProduct(self, filePath: Path, *, fileType: str | None = None) -> None:
        self._productAPI.saveProduct(self._productIndex, filePath, fileType=fileType)


class ConcreteWorkflowAPI(WorkflowAPI):
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        patternsAPI: PatternsAPI,
        productAPI: ProductAPI,
        scanAPI: ScanAPI,
        probeAPI: ProbeAPI,
        objectAPI: ObjectAPI,
        reconstructorAPI: ReconstructorAPI,
        executor: WorkflowExecutor,
    ) -> None:
        self._settingsRegistry = settingsRegistry
        self._patternsAPI = patternsAPI
        self._productAPI = productAPI
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._reconstructorAPI = reconstructorAPI
        self._executor = executor

    def _createProductAPI(self, productIndex: int) -> WorkflowProductAPI:
        if productIndex < 0:
            raise ValueError(f'Bad product index ({productIndex=})!')

        return ConcreteWorkflowProductAPI(
            self._productAPI,
            self._scanAPI,
            self._probeAPI,
            self._objectAPI,
            self._reconstructorAPI,
            self._executor,
            productIndex,
        )

    def openPatterns(
        self,
        filePath: Path,
        *,
        fileType: str | None = None,
        cropCenter: CropCenter | None = None,
        cropExtent: ImageExtent | None = None,
    ) -> None:
        self._patternsAPI.openPatterns(
            filePath, fileType=fileType, cropCenter=cropCenter, cropExtent=cropExtent
        )

    def importAssembledPatterns(self, filePath: Path) -> None:
        self._patternsAPI.importAssembledPatterns(filePath)

    def exportAssembledPatterns(self, filePath: Path) -> None:
        self._patternsAPI.exportAssembledPatterns(filePath)

    def openProduct(self, filePath: Path, *, fileType: str | None = None) -> WorkflowProductAPI:
        productIndex = self._productAPI.openProduct(filePath, fileType=fileType)
        return self._createProductAPI(productIndex)

    def createProduct(
        self,
        name: str,
        *,
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonCount: float | None = None,
        exposureTimeInSeconds: float | None = None,
    ) -> WorkflowProductAPI:
        productIndex = self._productAPI.insertNewProduct(
            name,
            comments=comments,
            detectorDistanceInMeters=detectorDistanceInMeters,
            probeEnergyInElectronVolts=probeEnergyInElectronVolts,
            probePhotonCount=probePhotonCount,
            exposureTimeInSeconds=exposureTimeInSeconds,
        )
        return self._createProductAPI(productIndex)

    def saveSettings(
        self, filePath: Path, changePathPrefix: PathPrefixChange | None = None
    ) -> None:
        self._settingsRegistry.saveSettings(filePath, changePathPrefix)
