from pathlib import Path
import logging

from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..product import ObjectRepository, ProductRepository
from ..reconstructor import DiffractionPatternPositionMatcher
from ..visualization import VisualizationEngine
from .exposure import ExposureAnalyzer
from .fluorescence import FluorescenceEnhancer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator
from .settings import FluorescenceSettings, ProbePropagationSettings
from .stxm import STXMSimulator
from .xmcd import XMCDAnalyzer

logger = logging.getLogger(__name__)


class AnalysisCore:
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        dataMatcher: DiffractionPatternPositionMatcher,
        productRepository: ProductRepository,
        objectRepository: ObjectRepository,
        upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
        deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
        fluorescenceFileReaderChooser: PluginChooser[FluorescenceFileReader],
        fluorescenceFileWriterChooser: PluginChooser[FluorescenceFileWriter],
    ) -> None:
        self.stxmSimulator = STXMSimulator(dataMatcher)
        self.stxmVisualizationEngine = VisualizationEngine(isComplex=False)

        self._probePropagationSettings = ProbePropagationSettings(settingsRegistry)
        self.probePropagator = ProbePropagator(self._probePropagationSettings, productRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=False)
        self.exposureAnalyzer = ExposureAnalyzer(productRepository)
        self.exposureVisualizationEngine = VisualizationEngine(isComplex=False)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)

        self._fluorescenceSettings = FluorescenceSettings(settingsRegistry)
        self.fluorescenceEnhancer = FluorescenceEnhancer(
            self._fluorescenceSettings,
            productRepository,
            upscalingStrategyChooser,
            deconvolutionStrategyChooser,
            fluorescenceFileReaderChooser,
            fluorescenceFileWriterChooser,
            settingsRegistry,
        )
        self.fluorescenceVisualizationEngine = VisualizationEngine(isComplex=False)
        self.xmcdAnalyzer = XMCDAnalyzer(objectRepository)
        self.xmcdVisualizationEngine = VisualizationEngine(isComplex=False)

    def enhanceFluorescence(
        self, productIndex: int, inputFilePath: Path, outputFilePath: Path
    ) -> int:
        fileType = 'XRF-Maps'

        try:
            self.fluorescenceEnhancer.setProduct(productIndex)
            self.fluorescenceEnhancer.openMeasuredDataset(inputFilePath, fileType)
            self.fluorescenceEnhancer.enhanceFluorescence()
            self.fluorescenceEnhancer.saveEnhancedDataset(outputFilePath, fileType)
        except Exception as exc:
            logger.exception(exc)
            return -1

        return 0
