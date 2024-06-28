from ptychodus.api.fluorescence import (DeconvolutionStrategy, FluorescenceFileReader,
                                        FluorescenceFileWriter, UpscalingStrategy)
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


class AnalysisCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 dataMatcher: DiffractionPatternPositionMatcher,
                 productRepository: ProductRepository, objectRepository: ObjectRepository,
                 upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
                 deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
                 fluorescenceFileReaderChooser: PluginChooser[FluorescenceFileReader],
                 fluorescenceFileWriterChooser: PluginChooser[FluorescenceFileWriter]) -> None:
        self.stxmSimulator = STXMSimulator(dataMatcher)
        self.stxmVisualizationEngine = VisualizationEngine(isComplex=False)

        self._probePropagationSettings = ProbePropagationSettings(settingsRegistry)
        self.probePropagator = ProbePropagator(self._probePropagationSettings, productRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=True)
        self.exposureAnalyzer = ExposureAnalyzer(productRepository)
        self.exposureVisualizationEngine = VisualizationEngine(isComplex=False)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)

        self._fluorescenceSettings = FluorescenceSettings(settingsRegistry)
        self.fluorescenceEnhancer = FluorescenceEnhancer(self._fluorescenceSettings, dataMatcher,
                                                         upscalingStrategyChooser,
                                                         deconvolutionStrategyChooser,
                                                         fluorescenceFileReaderChooser,
                                                         fluorescenceFileWriterChooser)
        self.fluorescenceVisualizationEngine = VisualizationEngine(isComplex=False)
        self.xmcdAnalyzer = XMCDAnalyzer(objectRepository)
        self.xmcdVisualizationEngine = VisualizationEngine(isComplex=False)
