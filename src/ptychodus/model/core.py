from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import overload
import logging
import sys

try:
    # NOTE must import hdf5plugin before h5py
    import hdf5plugin  # noqa
except ModuleNotFoundError:
    pass

import h5py
import numpy

from ptychodus.api.patterns import DiffractionMetadata, DiffractionPatternArray
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.workflow import WorkflowAPI

from .agent import AgentCore, AgentPresenter, ArgoSettings, ChatRepository
from .analysis import (
    AnalysisCore,
    ExposureAnalyzer,
    FourierRingCorrelator,
    ProbePropagator,
    STXMSimulator,
    XMCDAnalyzer,
)
from .automation import (
    AutomationCore,
    AutomationPresenter,
    AutomationProcessingPresenter,
)
from .fluorescence import FluorescenceCore, FluorescenceEnhancer
from .memory import MemoryPresenter
from .patterns import (
    Detector,
    DiffractionDatasetInputOutputPresenter,
    DiffractionDatasetPresenter,
    DiffractionMetadataPresenter,
    DiffractionPatternPresenter,
    PatternsCore,
)
from .product import (
    ObjectAPI,
    ObjectRepository,
    ProbeAPI,
    ProbeRepository,
    ProductAPI,
    ProductCore,
    ProductRepository,
    ScanAPI,
    ScanRepository,
)
from .ptychi import PtyChiReconstructorLibrary
from .ptychonn import PtychoNNReconstructorLibrary
from .reconstructor import ReconstructorCore, ReconstructorPresenter
from .tike import TikeReconstructorLibrary
from .visualization import VisualizationEngine
from .workflow import (
    WorkflowAuthorizationPresenter,
    WorkflowCore,
    WorkflowExecutionPresenter,
    WorkflowParametersPresenter,
    WorkflowStatusPresenter,
)

logger = logging.getLogger(__name__)


def configureLogger(isDeveloperModeEnabled: bool) -> None:
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        stream=sys.stdout,
        encoding='utf-8',
        level=logging.DEBUG if isDeveloperModeEnabled else logging.INFO,
    )
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger.info(f'Ptychodus {version("ptychodus")}')
    logger.info(f'NumPy {version("numpy")}')
    logger.info(f'Matplotlib {version("matplotlib")}')
    logger.info(f'HDF5Plugin {version("hdf5plugin")}')
    logger.info(f'H5Py {version("h5py")}')
    logger.info(f'HDF5 {h5py.version.hdf5_version}')


class ModelCore:
    def __init__(
        self, settingsFile: Path | None = None, *, isDeveloperModeEnabled: bool = False
    ) -> None:
        configureLogger(isDeveloperModeEnabled)
        self.rng = numpy.random.default_rng()
        self._pluginRegistry = PluginRegistry.loadPlugins()

        self.memoryPresenter = MemoryPresenter()
        self.settingsRegistry = SettingsRegistry()

        self._patternsCore = PatternsCore(
            self.settingsRegistry,
            self._pluginRegistry.diffractionFileReaders,
            self._pluginRegistry.diffractionFileWriters,
        )
        self._productCore = ProductCore(
            self.rng,
            self.settingsRegistry,
            self._patternsCore.detector,
            self._patternsCore.productSettings,
            self._patternsCore.patternSizer,
            self._patternsCore.dataset,
            self._pluginRegistry.scanFileReaders,
            self._pluginRegistry.scanFileWriters,
            self._pluginRegistry.fresnelZonePlates,
            self._pluginRegistry.probeFileReaders,
            self._pluginRegistry.probeFileWriters,
            self._pluginRegistry.objectFileReaders,
            self._pluginRegistry.objectFileWriters,
            self._pluginRegistry.productFileReaders,
            self._pluginRegistry.productFileWriters,
            self.settingsRegistry,
        )

        self.patternVisualizationEngine = VisualizationEngine(isComplex=False)
        self.probeVisualizationEngine = VisualizationEngine(isComplex=True)
        self.objectVisualizationEngine = VisualizationEngine(isComplex=True)

        self.ptyChiReconstructorLibrary = PtyChiReconstructorLibrary(
            self.settingsRegistry, self.detector, isDeveloperModeEnabled
        )
        self.tikeReconstructorLibrary = TikeReconstructorLibrary.createInstance(
            self.settingsRegistry, isDeveloperModeEnabled
        )
        self.ptychonnReconstructorLibrary = PtychoNNReconstructorLibrary.createInstance(
            self.settingsRegistry, isDeveloperModeEnabled
        )
        self._reconstructorCore = ReconstructorCore(
            self.settingsRegistry,
            self._patternsCore.dataset,
            self._productCore.productRepository,
            [
                self.ptyChiReconstructorLibrary,
                self.tikeReconstructorLibrary,
                self.ptychonnReconstructorLibrary,
            ],
        )
        self._fluorescenceCore = FluorescenceCore(
            self.settingsRegistry,
            self._productCore.productRepository,
            self._pluginRegistry.upscalingStrategies,
            self._pluginRegistry.deconvolutionStrategies,
            self._pluginRegistry.fluorescenceFileReaders,
            self._pluginRegistry.fluorescenceFileWriters,
        )
        self._analysisCore = AnalysisCore(
            self.settingsRegistry,
            self._reconstructorCore.dataMatcher,
            self._productCore.productRepository,
            self._productCore.objectRepository,
        )
        self._workflowCore = WorkflowCore(
            self.settingsRegistry,
            self._patternsCore.patternsAPI,
            self._productCore.productAPI,
            self._productCore.scanAPI,
            self._productCore.probeAPI,
            self._productCore.objectAPI,
            self._reconstructorCore.reconstructorAPI,
        )
        self._automationCore = AutomationCore(
            self.settingsRegistry,
            self._workflowCore.workflowAPI,
            self._pluginRegistry.fileBasedWorkflows,
        )
        self._agentCore = AgentCore(self.settingsRegistry)

        if settingsFile:
            self.settingsRegistry.openSettings(settingsFile)

    def __enter__(self) -> ModelCore:
        self._patternsCore.start()
        self._reconstructorCore.start()
        self._workflowCore.start()
        self._automationCore.start()
        return self

    @overload
    def __exit__(self, exception_type: None, exception_value: None, traceback: None) -> None: ...

    @overload
    def __exit__(
        self,
        exception_type: type[BaseException],
        exception_value: BaseException,
        traceback: TracebackType,
    ) -> None: ...

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._automationCore.stop()
        self._workflowCore.stop()
        self._reconstructorCore.stop()
        self._patternsCore.stop()

    @property
    def diffractionDatasetInputOutputPresenter(
        self,
    ) -> DiffractionDatasetInputOutputPresenter:
        return self._patternsCore.datasetInputOutputPresenter

    @property
    def diffractionMetadataPresenter(self) -> DiffractionMetadataPresenter:
        return self._patternsCore.metadataPresenter

    @property
    def diffractionDatasetPresenter(self) -> DiffractionDatasetPresenter:
        return self._patternsCore.datasetPresenter

    @property
    def patternPresenter(self) -> DiffractionPatternPresenter:
        return self._patternsCore.patternPresenter

    @property
    def productRepository(self) -> ProductRepository:
        return self._productCore.productRepository

    @property
    def productAPI(self) -> ProductAPI:
        return self._productCore.productAPI

    @property
    def scanRepository(self) -> ScanRepository:
        return self._productCore.scanRepository

    @property
    def scanAPI(self) -> ScanAPI:
        return self._productCore.scanAPI

    @property
    def probeRepository(self) -> ProbeRepository:
        return self._productCore.probeRepository

    @property
    def probeAPI(self) -> ProbeAPI:
        return self._productCore.probeAPI

    @property
    def objectRepository(self) -> ObjectRepository:
        return self._productCore.objectRepository

    @property
    def objectAPI(self) -> ObjectAPI:
        return self._productCore.objectAPI

    def initializeStreamingWorkflow(self, metadata: DiffractionMetadata) -> None:
        self._patternsCore.patternsAPI.initializeStreaming(metadata)
        self._patternsCore.patternsAPI.startAssemblingDiffractionPatterns()
        self._productCore.scanAPI.initializeStreamingScan()  # FIXME

    def assembleDiffractionPattern(self, array: DiffractionPatternArray, timeStamp: float) -> None:
        self._patternsCore.patternsAPI.assemble(array)
        self._productCore.scanAPI.insertArrayTimeStamp(array.getIndex(), timeStamp)  # FIXME

    def assembleScanPositionsX(
        self, valuesInMeters: Sequence[float], timeStamps: Sequence[float]
    ) -> None:
        self._productCore.scanAPI.assembleScanPositionsX(valuesInMeters, timeStamps)  # FIXME

    def assembleScanPositionsY(
        self, valuesInMeters: Sequence[float], timeStamps: Sequence[float]
    ) -> None:
        self._productCore.scanAPI.assembleScanPositionsY(valuesInMeters, timeStamps)  # FIXME

    def finalizeStreamingWorkflow(self) -> None:
        self._productCore.scanAPI.finalizeStreamingScan()  # FIXME
        self._patternsCore.patternsAPI.stopAssemblingDiffractionPatterns(finishAssembling=True)

    def getDiffractionPatternAssemblyQueueSize(self) -> int:
        return self._patternsCore.patternsAPI.getAssemblyQueueSize()

    def refreshActiveDataset(self) -> None:
        self._patternsCore.dataset.notifyObserversIfDatasetChanged()

    def batchModeExecute(
        self,
        action: str,
        inputPath: Path,
        outputPath: Path,
        *,
        productFileType: str = 'NPZ',
        fluorescenceInputFilePath: Path | None = None,
        fluorescenceOutputFilePath: Path | None = None,
    ) -> int:
        # TODO add enum for actions; implement using workflow API
        if action.lower() == 'train':
            output = self._reconstructorCore.reconstructorAPI.train(inputPath)
            self._reconstructorCore.reconstructorAPI.saveModel(outputPath)
            return output.result

        inputProductIndex = self._productCore.productAPI.openProduct(
            inputPath, fileType=productFileType
        )

        if inputProductIndex < 0:
            logger.error(f'Failed to open product "{inputPath}"!')
            return -1

        if action.lower() == 'reconstruct':
            logger.info('Reconstructing...')
            outputProductIndex = self._reconstructorCore.reconstructorAPI.reconstruct(
                inputProductIndex
            )
            self._reconstructorCore.reconstructorAPI.processResults(block=True)
            logger.info('Reconstruction complete.')

            self._productCore.productAPI.saveProduct(
                outputProductIndex, outputPath, fileType=productFileType
            )

            if fluorescenceInputFilePath is not None and fluorescenceOutputFilePath is not None:
                self._fluorescenceCore.enhanceFluorescence(
                    outputProductIndex,
                    fluorescenceInputFilePath,
                    fluorescenceOutputFilePath,
                )
        elif action.lower() == 'prepare_training_data':
            self._reconstructorCore.reconstructorAPI.exportTrainingData(
                outputPath, inputProductIndex
            )
        else:
            logger.error(f'Unknown batch mode action "{action}"!')
            return -1

        return 0

    @property
    def detector(self) -> Detector:
        return self._patternsCore.detector

    @property
    def reconstructorPresenter(self) -> ReconstructorPresenter:
        return self._reconstructorCore.presenter

    @property
    def stxmSimulator(self) -> STXMSimulator:
        return self._analysisCore.stxmSimulator

    @property
    def stxmVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.stxmVisualizationEngine

    @property
    def probePropagator(self) -> ProbePropagator:
        return self._analysisCore.probePropagator

    @property
    def probePropagatorVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.probePropagatorVisualizationEngine

    @property
    def exposureAnalyzer(self) -> ExposureAnalyzer:
        return self._analysisCore.exposureAnalyzer

    @property
    def exposureVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.exposureVisualizationEngine

    @property
    def fourierRingCorrelator(self) -> FourierRingCorrelator:
        return self._analysisCore.fourierRingCorrelator

    @property
    def fluorescenceEnhancer(self) -> FluorescenceEnhancer:
        return self._fluorescenceCore.enhancer

    @property
    def fluorescenceVisualizationEngine(self) -> VisualizationEngine:
        return self._fluorescenceCore.visualizationEngine

    @property
    def xmcdAnalyzer(self) -> XMCDAnalyzer:
        return self._analysisCore.xmcdAnalyzer

    @property
    def xmcdVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.xmcdVisualizationEngine

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
    def workflowAPI(self) -> WorkflowAPI:
        return self._workflowCore.workflowAPI

    @property
    def automationPresenter(self) -> AutomationPresenter:
        return self._automationCore.presenter

    @property
    def automationProcessingPresenter(self) -> AutomationProcessingPresenter:
        return self._automationCore.processingPresenter

    @property
    def argoSettings(self) -> ArgoSettings:
        return self._agentCore.argoSettings

    @property
    def agentPresenter(self) -> AgentPresenter:
        return self._agentCore.presenter

    @property
    def chatRepository(self) -> ChatRepository:
        return self._agentCore.chatRepository
