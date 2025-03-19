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

from .analysis import (
    AnalysisCore,
    IlluminationMapper,
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
from .metadata import MetadataPresenter
from .patterns import PatternsCore, PatternsStreamingContext
from .product import (
    ObjectAPI,
    ObjectRepository,
    PositionsStreamingContext,
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


class PtychodusStreamingContext:
    def __init__(
        self,
        positions_context: PositionsStreamingContext,
        patterns_context: PatternsStreamingContext,
    ) -> None:
        self._positions_context = positions_context
        self._patterns_context = patterns_context

    def start(self) -> None:
        self._positions_context.start()
        self._patterns_context.start()

    def append_positions_x(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_context.append_positions_x(values_m, trigger_counts)

    def append_positions_y(self, values_m: Sequence[float], trigger_counts: Sequence[int]) -> None:
        self._positions_context.append_positions_y(values_m, trigger_counts)

    def append_array(self, array: DiffractionPatternArray) -> None:
        self._patterns_context.append_array(array)

    def get_queue_size(self) -> int:
        return self._patterns_context.get_queue_size()

    def stop(self) -> None:
        self._patterns_context.stop()
        self._positions_context.stop()


class ModelCore:
    def __init__(
        self, settingsFile: Path | None = None, *, is_developer_mode_enabled: bool = False
    ) -> None:
        configureLogger(is_developer_mode_enabled)
        self.rng = numpy.random.default_rng()
        self._pluginRegistry = PluginRegistry.load_plugins()

        self.memoryPresenter = MemoryPresenter()
        self.settingsRegistry = SettingsRegistry()

        self.patterns_core = PatternsCore(
            self.settingsRegistry,
            self._pluginRegistry.diffraction_file_readers,
            self._pluginRegistry.diffraction_file_writers,
            self.settingsRegistry,
        )
        self._productCore = ProductCore(
            self.rng,
            self.settingsRegistry,
            self.patterns_core.patternSizer,
            self.patterns_core.dataset,
            self._pluginRegistry.position_file_readers,
            self._pluginRegistry.position_file_writers,
            self._pluginRegistry.fresnel_zone_plates,
            self._pluginRegistry.probe_file_readers,
            self._pluginRegistry.probe_file_writers,
            self._pluginRegistry.object_file_readers,
            self._pluginRegistry.object_file_writers,
            self._pluginRegistry.product_file_readers,
            self._pluginRegistry.product_file_writers,
            self.settingsRegistry,
        )
        self.metadataPresenter = MetadataPresenter(
            self.patterns_core.detectorSettings,
            self.patterns_core.patternSettings,
            self.patterns_core.dataset,
            self._productCore.settings,
        )

        self.patternVisualizationEngine = VisualizationEngine(is_complex=False)
        self.probeVisualizationEngine = VisualizationEngine(is_complex=True)
        self.objectVisualizationEngine = VisualizationEngine(is_complex=True)

        self.ptyChiReconstructorLibrary = PtyChiReconstructorLibrary(
            self.settingsRegistry, self.patterns_core.patternSizer, is_developer_mode_enabled
        )
        self.tikeReconstructorLibrary = TikeReconstructorLibrary.create_instance(
            self.settingsRegistry, is_developer_mode_enabled
        )
        self.ptychonnReconstructorLibrary = PtychoNNReconstructorLibrary.create_instance(
            self.settingsRegistry, is_developer_mode_enabled
        )
        self._reconstructorCore = ReconstructorCore(
            self.settingsRegistry,
            self.patterns_core.dataset,
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
            self._pluginRegistry.upscaling_strategies,
            self._pluginRegistry.deconvolution_strategies,
            self._pluginRegistry.fluorescence_file_readers,
            self._pluginRegistry.fluorescence_file_writers,
        )
        self._analysisCore = AnalysisCore(
            self.settingsRegistry,
            self._reconstructorCore.dataMatcher,
            self._productCore.productRepository,
            self._productCore.objectRepository,
        )
        self._workflowCore = WorkflowCore(
            self.settingsRegistry,
            self.patterns_core.patternsAPI,
            self._productCore.productAPI,
            self._productCore.scanAPI,
            self._productCore.probeAPI,
            self._productCore.objectAPI,
            self._reconstructorCore.reconstructorAPI,
        )
        self._automationCore = AutomationCore(
            self.settingsRegistry,
            self._workflowCore.workflowAPI,
            self._pluginRegistry.file_based_workflows,
        )

        if settingsFile:
            self.settingsRegistry.open_settings(settingsFile)

    def __enter__(self) -> ModelCore:
        self.patterns_core.start()
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
        self.patterns_core.stop()

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

    def createStreamingContext(self, metadata: DiffractionMetadata) -> PtychodusStreamingContext:
        return PtychodusStreamingContext(
            self._productCore.scanAPI.createStreamingContext(),
            self.patterns_core.patternsAPI.createStreamingContext(metadata),
        )

    def refreshActiveDataset(self) -> None:
        self.patterns_core.dataset.assemble_patterns()

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
            inputPath, file_type=productFileType
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
                outputProductIndex, outputPath, file_type=productFileType
            )

            if fluorescenceInputFilePath is not None and fluorescenceOutputFilePath is not None:
                self._fluorescenceCore.enhance_fluorescence(
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
    def reconstructorPresenter(self) -> ReconstructorPresenter:
        return self._reconstructorCore.presenter

    @property
    def stxmSimulator(self) -> STXMSimulator:
        return self._analysisCore.stxm_simulator

    @property
    def stxmVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.stxm_visualization_engine

    @property
    def probePropagator(self) -> ProbePropagator:
        return self._analysisCore.probe_propagator

    @property
    def probePropagatorVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.probe_propagator_visualization_engine

    @property
    def exposureAnalyzer(self) -> IlluminationMapper:
        return self._analysisCore.exposure_analyzer

    @property
    def exposureVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.exposure_visualization_engine

    @property
    def fourierRingCorrelator(self) -> FourierRingCorrelator:
        return self._analysisCore.fourier_ring_correlator

    @property
    def fluorescenceEnhancer(self) -> FluorescenceEnhancer:
        return self._fluorescenceCore.enhancer

    @property
    def fluorescenceVisualizationEngine(self) -> VisualizationEngine:
        return self._fluorescenceCore.visualization_engine

    @property
    def xmcdAnalyzer(self) -> XMCDAnalyzer:
        return self._analysisCore.xmcd_analyzer

    @property
    def xmcdVisualizationEngine(self) -> VisualizationEngine:
        return self._analysisCore.xmcd_visualization_engine

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
    def workflow_api(self) -> WorkflowAPI:
        return self._workflowCore.workflowAPI

    @property
    def automationPresenter(self) -> AutomationPresenter:
        return self._automationCore.presenter

    @property
    def automationProcessingPresenter(self) -> AutomationProcessingPresenter:
        return self._automationCore.processingPresenter
