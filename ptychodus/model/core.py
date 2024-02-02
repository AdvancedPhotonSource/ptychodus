from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import overload
import logging
import sys

try:
    # NOTE must import hdf5plugin before h5py
    import hdf5plugin
except ModuleNotFoundError:
    pass

import h5py
import numpy

from ..api.patterns import DiffractionMetadata, DiffractionPatternArray
from ..api.plugins import PluginRegistry
from ..api.settings import SettingsRegistry
from ..api.state import StateDataRegistry
from .automation import AutomationCore, AutomationPresenter, AutomationProcessingPresenter
from .image import ImageCore, ImagePresenter
from .memory import MemoryPresenter
from .metadata import MetadataCore
from .object import ObjectCore
from .patterns import (DetectorPresenter, DiffractionDatasetInputOutputPresenter,
                       DiffractionDatasetPresenter, DiffractionPatternPresenter, PatternsCore)
from .probe import ProbeCore
from .product import ProductCore
from .ptychonn import PtychoNNReconstructorLibrary
from .ptychopy import PtychoPyReconstructorLibrary
from .reconstructor import ReconstructorCore, ReconstructorPresenter
from .rpc import RPCMessageService
from .rpcLoadResults import LoadResultsExecutor, LoadResultsMessage
from .scan import ScanCore
from .tike import TikeReconstructorLibrary
from .workflow import (WorkflowAuthorizationPresenter, WorkflowCore, WorkflowExecutionPresenter,
                       WorkflowParametersPresenter, WorkflowStatusPresenter)

logger = logging.getLogger(__name__)


def configureLogger() -> None:
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                        stream=sys.stdout,
                        encoding='utf-8',
                        level=logging.DEBUG)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tike').setLevel(logging.INFO)

    logger.info(f'Ptychodus {version("ptychodus")}')
    logger.info(f'NumPy {version("numpy")}')
    logger.info(f'Matplotlib {version("matplotlib")}')
    logger.info(f'HDF5Plugin {version("hdf5plugin")}')
    logger.info(f'H5Py {version("h5py")}')
    logger.info(f'HDF5 {h5py.version.hdf5_version}')


@dataclass(frozen=True)
class ModelArgs:
    restartFilePath: Path | None
    settingsFilePath: Path | None
    replacementPathPrefix: str | None = None
    rpcPort: int = -1
    autoExecuteRPCs: bool = False
    isDeveloperModeEnabled: bool = False


class ModelCore:

    def __init__(self, modelArgs: ModelArgs) -> None:
        configureLogger()
        self._modelArgs = modelArgs
        self.rng = numpy.random.default_rng()
        self._pluginRegistry = PluginRegistry.loadPlugins()

        self.memoryPresenter = MemoryPresenter()
        self.settingsRegistry = SettingsRegistry(modelArgs.replacementPathPrefix)
        self._patternsCore = PatternsCore(self.settingsRegistry,
                                          self._pluginRegistry.diffractionFileReaders)
        self._detectorImageCore = ImageCore(self._pluginRegistry.scalarTransformations,
                                            isComplex=False)

        self._metadataCore = MetadataCore(self.settingsRegistry, self._patternsCore.dataset,
                                          self._patternsCore.detector,
                                          self._patternsCore.patternSettings)
        self._scanCore = ScanCore(self.rng, self._pluginRegistry.scanFileReaders,
                                  self._pluginRegistry.scanFileWriters)
        self._probeCore = ProbeCore(self.rng, self._pluginRegistry.probeFileReaders,
                                    self._pluginRegistry.probeFileWriters)
        self._probeImageCore = ImageCore(self._pluginRegistry.scalarTransformations.copy(),
                                         isComplex=True)
        self._objectCore = ObjectCore(self.rng, self._pluginRegistry.objectFileReaders,
                                      self._pluginRegistry.objectFileWriters)
        self._objectImageCore = ImageCore(self._pluginRegistry.scalarTransformations.copy(),
                                          isComplex=True)
        self._productCore = ProductCore(
            self._patternsCore.patternSizer, self._patternsCore.dataset,
            self._metadataCore.builder, self._scanCore.repositoryItemFactory,
            self._scanCore.builderFactory, self._probeCore.repositoryItemFactory,
            self._probeCore.builderFactory, self._objectCore.repositoryItemFactory,
            self._objectCore.builderFactory, self._pluginRegistry.productFileReaders,
            self._pluginRegistry.productFileWriters)

        self.tikeReconstructorLibrary = TikeReconstructorLibrary.createInstance(
            self.settingsRegistry, modelArgs.isDeveloperModeEnabled)
        self.ptychonnReconstructorLibrary = PtychoNNReconstructorLibrary.createInstance(
            self.settingsRegistry, self._objectCore.objectAPI, modelArgs.isDeveloperModeEnabled)
        self.ptychopyReconstructorLibrary = PtychoPyReconstructorLibrary.createInstance(
            self.settingsRegistry, modelArgs.isDeveloperModeEnabled)
        self._reconstructorCore = ReconstructorCore(
            self.settingsRegistry,
            self._patternsCore.dataset,
            self._productCore.productRepository,
            [
                self.tikeReconstructorLibrary,
                self.ptychonnReconstructorLibrary,
                self.ptychopyReconstructorLibrary,
            ],
        )

        self._stateDataRegistry = StateDataRegistry(self._patternsCore)
        self._workflowCore = WorkflowCore(self.settingsRegistry, self._stateDataRegistry)
        self._automationCore = AutomationCore(self.settingsRegistry, self._patternsCore.dataAPI,
                                              self._scanCore.scanAPI, self._probeCore.probeAPI,
                                              self._objectCore.objectAPI, self._workflowCore)

        self.rpcMessageService: RPCMessageService | None = None

        if modelArgs.rpcPort >= 0:
            self.rpcMessageService = RPCMessageService(modelArgs.rpcPort,
                                                       modelArgs.autoExecuteRPCs)
            self.rpcMessageService.registerProcedure(LoadResultsMessage,
                                                     LoadResultsExecutor(self._stateDataRegistry))

    def __enter__(self) -> ModelCore:
        if self._modelArgs.settingsFilePath:
            self.settingsRegistry.openSettings(self._modelArgs.settingsFilePath)

        if self._modelArgs.restartFilePath:
            self.openStateData(self._modelArgs.restartFilePath)

        if self._patternsCore.dataAPI.getAssemblyQueueSize() > 0:
            self._patternsCore.dataAPI.startAssemblingDiffractionPatterns()
            self._patternsCore.dataAPI.stopAssemblingDiffractionPatterns(finishAssembling=True)

        if self.rpcMessageService:
            self.rpcMessageService.start()

        self._patternsCore.start()
        self._workflowCore.start()
        self._automationCore.start()

        return self

    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None:
        ...

    @overload
    def __exit__(self, exception_type: type[BaseException], exception_value: BaseException,
                 traceback: TracebackType) -> None:
        ...

    def __exit__(self, exception_type: type[BaseException] | None,
                 exception_value: BaseException | None, traceback: TracebackType | None) -> None:
        self._automationCore.stop()
        self._workflowCore.stop()
        self._patternsCore.stop()

        if self.rpcMessageService:
            self.rpcMessageService.stop()

    @property
    def detectorImagePresenter(self) -> ImagePresenter:
        return self._detectorImageCore.presenter

    @property
    def patternPresenter(self) -> DiffractionPatternPresenter:
        return self._patternsCore.patternPresenter

    @property
    def diffractionDatasetPresenter(self) -> DiffractionDatasetPresenter:
        return self._patternsCore.datasetPresenter

    @property
    def diffractionDatasetInputOutputPresenter(self) -> DiffractionDatasetInputOutputPresenter:
        return self._patternsCore.datasetInputOutputPresenter

    def initializeStreamingWorkflow(self, metadata: DiffractionMetadata) -> None:
        self._patternsCore.dataAPI.initializeStreaming(metadata)
        self._patternsCore.dataAPI.startAssemblingDiffractionPatterns()
        self._scanCore.scanAPI.initializeStreamingScan()

    def assembleDiffractionPattern(self, array: DiffractionPatternArray, timeStamp: float) -> None:
        self._patternsCore.dataAPI.assemble(array)
        self._scanCore.scanAPI.insertArrayTimeStamp(array.getIndex(), timeStamp)

    def assembleScanPositionsX(self, valuesInMeters: Sequence[float],
                               timeStamps: Sequence[float]) -> None:
        self._scanCore.scanAPI.assembleScanPositionsX(valuesInMeters, timeStamps)

    def assembleScanPositionsY(self, valuesInMeters: Sequence[float],
                               timeStamps: Sequence[float]) -> None:
        self._scanCore.scanAPI.assembleScanPositionsY(valuesInMeters, timeStamps)

    def finalizeStreamingWorkflow(self) -> None:
        self._scanCore.scanAPI.finalizeStreamingScan()
        self._patternsCore.dataAPI.stopAssemblingDiffractionPatterns(finishAssembling=True)
        self._objectCore.objectAPI.selectNewItemFromInitializerSimpleName('Random')

    def getDiffractionPatternAssemblyQueueSize(self) -> int:
        return self._patternsCore.dataAPI.getAssemblyQueueSize()

    def refreshActiveDataset(self) -> None:
        self._patternsCore.dataset.notifyObserversIfDatasetChanged()

    def refreshAutomationDatasets(self) -> None:
        self._automationCore.repository.notifyObserversIfRepositoryChanged()

    def saveStateData(self, filePath: Path, *, restartable: bool) -> None:
        self._stateDataRegistry.saveStateData(filePath, restartable=restartable)

    def openStateData(self, filePath: Path) -> None:
        self._stateDataRegistry.openStateData(filePath)

    def batchModeReconstruct(self, filePath: Path) -> int:
        result = self._reconstructorCore.reconstructorAPI.reconstruct()
        self.saveStateData(filePath, restartable=False)
        return result.result

    def batchModeTrain(self, directoryPath: Path) -> float:
        for filePath in directoryPath.glob('*.npz'):
            # TODO sort by filePath.stat().st_mtime
            self._stateDataRegistry.openStateData(filePath)
            self._reconstructorCore.reconstructorAPI.ingestTrainingData()

        self._reconstructorCore.reconstructorAPI.train()

        someQualityMetric = 0.  # TODO
        return someQualityMetric

    @property
    def detectorPresenter(self) -> DetectorPresenter:
        return self._patternsCore.detectorPresenter

    @property
    def probeImagePresenter(self) -> ImagePresenter:
        return self._probeImageCore.presenter

    @property
    def objectImagePresenter(self) -> ImagePresenter:
        return self._objectImageCore.presenter

    @property
    def reconstructorPresenter(self) -> ReconstructorPresenter:
        return self._reconstructorCore.presenter

    @property
    def areWorkflowsSupported(self) -> bool:
        return self._workflowCore.areWorkflowsSupported

    @property
    def workflowAuthorizationPresenter(self) -> WorkflowAuthorizationPresenter:
        return self._workflowCore.authorizationPresenter

    @property
    def workflowStatusPresenter(self) -> WorkflowStatusPresenter:
        return self._workflowCore.statusPresenter

    @property
    def workflowExecutionPresenter(self) -> WorkflowExecutionPresenter:
        return self._workflowCore.executionPresenter

    @property
    def workflowParametersPresenter(self) -> WorkflowParametersPresenter:
        return self._workflowCore.parametersPresenter

    @property
    def automationPresenter(self) -> AutomationPresenter:
        return self._automationCore.presenter

    @property
    def automationProcessingPresenter(self) -> AutomationProcessingPresenter:
        return self._automationCore.processingPresenter
