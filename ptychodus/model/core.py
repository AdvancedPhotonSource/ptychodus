from __future__ import annotations
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import overload, Optional
import logging
import sys

import h5py
import matplotlib
import numpy

from ..api.data import DiffractionMetadata, DiffractionPatternArray
from ..api.plugins import PluginRegistry
from ..api.settings import SettingsRegistry
from .automation import AutomationCore, AutomationPresenter, AutomationProcessingPresenter
from .data import (ActiveDiffractionPatternPresenter, DataCore, DiffractionDatasetPresenter,
                   DiffractionDatasetInputOutputPresenter, DiffractionPatternPresenter)
from .detector import Detector, DetectorPresenter, DetectorSettings
from .image import ImageCore, ImagePresenter
from .metadata import MetadataPresenter
from .object import ObjectCore, ObjectPresenter, ObjectRepositoryPresenter
from .probe import ApparatusPresenter, ProbeCore, ProbePresenter
from .ptychonn import PtychoNNReconstructorLibrary
from .ptychopy import PtychoPyReconstructorLibrary
from .reconstructor import ReconstructorCore, ReconstructorPresenter, ReconstructorPlotPresenter
from .rpc import RPCMessageService
from .rpcLoadResults import LoadResultsExecutor, LoadResultsMessage
from .scan import ScanCore, ScanPresenter, ScanRepositoryPresenter
from .statefulCore import StateDataRegistry
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

    logger.info(f'Ptychodus {version("ptychodus")}')
    logger.info(f'NumPy {version("numpy")}')
    logger.info(f'Matplotlib {version("matplotlib")}')
    logger.info(f'H5Py {version("h5py")}')
    logger.info(f'HDF5 {h5py.version.hdf5_version}')


@dataclass(frozen=True)
class ModelArgs:
    restartFilePath: Optional[Path]
    settingsFilePath: Optional[Path]
    replacementPathPrefix: Optional[str] = None
    rpcPort: int = -1
    autoExecuteRPCs: bool = False
    isDeveloperModeEnabled: bool = False


class ModelCore:

    def __init__(self, modelArgs: ModelArgs) -> None:
        configureLogger()
        self._modelArgs = modelArgs
        self.rng = numpy.random.default_rng()
        self._pluginRegistry = PluginRegistry.loadPlugins()

        self.settingsRegistry = SettingsRegistry(modelArgs.replacementPathPrefix)
        self._detectorSettings = DetectorSettings.createInstance(self.settingsRegistry)
        self._detector = Detector.createInstance(self._detectorSettings)
        self.detectorPresenter = DetectorPresenter.createInstance(self._detectorSettings)
        self._detectorImageCore = ImageCore(
            self._pluginRegistry.buildScalarTransformationChooser())

        self._dataCore = DataCore(self.settingsRegistry, self._detector,
                                  self._pluginRegistry.buildDiffractionFileReaderChooser())
        self._scanCore = ScanCore(self.rng, self.settingsRegistry, self._dataCore.dataset,
                                  self._pluginRegistry.buildScanFileReaderChooser(),
                                  self._pluginRegistry.buildScanFileWriterChooser())
        self._probeCore = ProbeCore(self.rng, self.settingsRegistry, self._detector,
                                    self._dataCore.patternSizer,
                                    self._pluginRegistry.buildProbeFileReaderChooser(),
                                    self._pluginRegistry.buildProbeFileWriterChooser())
        self._probeImageCore = ImageCore(self._pluginRegistry.buildScalarTransformationChooser())
        self._objectCore = ObjectCore(self.rng, self.settingsRegistry, self._probeCore.apparatus,
                                      self._scanCore.sizer, self._probeCore.sizer,
                                      self._pluginRegistry.buildObjectFileReaderChooser(),
                                      self._pluginRegistry.buildObjectFileWriterChooser())
        self._objectImageCore = ImageCore(self._pluginRegistry.buildScalarTransformationChooser())
        self.metadataPresenter = MetadataPresenter.createInstance(self._dataCore.dataset,
                                                                  self._detectorSettings,
                                                                  self._dataCore.patternSettings,
                                                                  self._probeCore.settings,
                                                                  self._scanCore.scanAPI)

        self.tikeReconstructorLibrary = TikeReconstructorLibrary.createInstance(
            self.settingsRegistry, self._dataCore.dataset, self._scanCore.scan,
            self._scanCore.scanAPI, self._probeCore.probe, self._objectCore.objectAPI,
            modelArgs.isDeveloperModeEnabled)
        self.ptychonnReconstructorLibrary = PtychoNNReconstructorLibrary.createInstance(
            self.settingsRegistry, self._dataCore.dataset, self._scanCore.scan,
            self._probeCore.apparatus, self._objectCore.objectAPI,
            modelArgs.isDeveloperModeEnabled)
        self.ptychopyReconstructorLibrary = PtychoPyReconstructorLibrary.createInstance(
            self.settingsRegistry, modelArgs.isDeveloperModeEnabled)
        self._reconstructorCore = ReconstructorCore(
            self.settingsRegistry,
            [
                self.tikeReconstructorLibrary,
                self.ptychonnReconstructorLibrary,
                self.ptychopyReconstructorLibrary,
            ],
        )

        self._stateDataRegistry = StateDataRegistry(
            (self._dataCore, self._scanCore, self._probeCore, self._objectCore))
        self._workflowCore = WorkflowCore(self.settingsRegistry, self._stateDataRegistry)
        self._automationCore = AutomationCore(self.settingsRegistry, self._dataCore,
                                              self._scanCore.scanAPI, self._probeCore,
                                              self._objectCore.objectAPI, self._workflowCore)

        self.rpcMessageService: Optional[RPCMessageService] = None

        if modelArgs.rpcPort >= 0:
            self.rpcMessageService = RPCMessageService(modelArgs.rpcPort,
                                                       modelArgs.autoExecuteRPCs)
            self.rpcMessageService.registerProcedure(
                LoadResultsMessage,
                LoadResultsExecutor(self._probeCore.probe, self._objectCore.objectAPI))

    def __enter__(self) -> ModelCore:
        if self._modelArgs.settingsFilePath:
            self.settingsRegistry.openSettings(self._modelArgs.settingsFilePath)

        if self._modelArgs.restartFilePath:
            self.openStateData(self._modelArgs.restartFilePath)

        if self.diffractionDatasetInputOutputPresenter.isReadyToAssemble:
            self.diffractionDatasetInputOutputPresenter.startProcessingDiffractionPatterns()
            self.diffractionDatasetInputOutputPresenter.stopProcessingDiffractionPatterns(
                finishAssembling=True)

        if self.rpcMessageService:
            self.rpcMessageService.start()

        self._dataCore.start()
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
        self._dataCore.stop()

        if self.rpcMessageService:
            self.rpcMessageService.stop()

    @property
    def detectorImagePresenter(self) -> ImagePresenter:
        return self._detectorImageCore.presenter

    @property
    def apparatusPresenter(self) -> ApparatusPresenter:
        return self._probeCore.apparatusPresenter

    @property
    def patternPresenter(self) -> DiffractionPatternPresenter:
        return self._dataCore.patternPresenter

    @property
    def activeDiffractionPatternPresenter(self) -> ActiveDiffractionPatternPresenter:
        return self._dataCore.activePatternPresenter

    @property
    def diffractionDatasetPresenter(self) -> DiffractionDatasetPresenter:
        return self._dataCore.datasetPresenter

    @property
    def diffractionDatasetInputOutputPresenter(self) -> DiffractionDatasetInputOutputPresenter:
        return self._dataCore.datasetInputOutputPresenter

    def initializeStreamingWorkflow(self, metadata: DiffractionMetadata) -> None:
        self.diffractionDatasetInputOutputPresenter.initializeStreaming(metadata)
        self.diffractionDatasetInputOutputPresenter.startProcessingDiffractionPatterns()
        self._scanCore.scanAPI.initializeStreamingScan()

    def assembleDiffractionPattern(self, array: DiffractionPatternArray, timeStamp: float) -> None:
        self.diffractionDatasetInputOutputPresenter.assemble(array)
        self._scanCore.scanAPI.insertArrayTimeStamp(array.getIndex(), timeStamp)

    def assembleScanPositionsX(self, valuesInMeters: list[float], timeStamps: list[float]) -> None:
        self._scanCore.scanAPI.assembleScanPositionsX(valuesInMeters, timeStamps)

    def assembleScanPositionsY(self, valuesInMeters: list[float], timeStamps: list[float]) -> None:
        self._scanCore.scanAPI.assembleScanPositionsY(valuesInMeters, timeStamps)

    def finalizeStreamingWorkflow(self) -> None:
        self.diffractionDatasetInputOutputPresenter.stopProcessingDiffractionPatterns(
            finishAssembling=True)
        self._scanCore.scanAPI.finalizeStreamingScan()
        self._objectCore.objectAPI.initializeAndActivateObject('Random')

    def getDiffractionPatternAssemblyQueueSize(self) -> int:
        return self.diffractionDatasetInputOutputPresenter.getAssemblyQueueSize()

    def refreshActiveDataset(self) -> None:
        self._dataCore.dataset.notifyObserversIfDatasetChanged()

    def refreshAutomationDatasets(self) -> None:
        self._automationCore.repository.notifyObserversIfRepositoryChanged()

    def saveStateData(self, filePath: Path, *, restartable: bool) -> None:
        self._stateDataRegistry.saveStateData(filePath, restartable=restartable)

    def openStateData(self, filePath: Path) -> None:
        self._stateDataRegistry.openStateData(filePath)

    def batchModeReconstruct(self, filePath: Path) -> int:
        result = self._reconstructorCore.presenter.reconstruct()
        self.saveStateData(filePath, restartable=False)
        return result.result

    @property
    def scanPresenter(self) -> ScanPresenter:
        return self._scanCore.presenter

    @property
    def scanRepositoryPresenter(self) -> ScanRepositoryPresenter:
        return self._scanCore.repositoryPresenter

    @property
    def probePresenter(self) -> ProbePresenter:
        return self._probeCore.presenter

    @property
    def probeImagePresenter(self) -> ImagePresenter:
        return self._probeImageCore.presenter

    @property
    def objectPresenter(self) -> ObjectPresenter:
        return self._objectCore.presenter

    @property
    def objectRepositoryPresenter(self) -> ObjectRepositoryPresenter:
        return self._objectCore.repositoryPresenter

    @property
    def objectImagePresenter(self) -> ImagePresenter:
        return self._objectImageCore.presenter

    @property
    def reconstructorPresenter(self) -> ReconstructorPresenter:
        return self._reconstructorCore.presenter

    @property
    def reconstructorPlotPresenter(self) -> ReconstructorPlotPresenter:
        return self._reconstructorCore.plotPresenter

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
