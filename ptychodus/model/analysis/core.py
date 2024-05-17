from ptychodus.api.fluorescence import (DeconvolutionStrategy, FluorescenceFileReader,
                                        FluorescenceFileWriter, UpscalingStrategy)
from ptychodus.api.plugins import PluginChooser

from ..product import ObjectRepository, ProductRepository
from ..visualization import VisualizationEngine
from .exposure import ExposureAnalyzer
from .fluorescence import FluorescenceEnhancer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator
from .stxm import STXMAnalyzer
from .xmcd import XMCDAnalyzer


class AnalysisCore:

    def __init__(self, productRepository: ProductRepository, objectRepository: ObjectRepository,
                 upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
                 deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
                 fluorescenceFileReaderChooser: PluginChooser[FluorescenceFileReader],
                 fluorescenceFileWriterChooser: PluginChooser[FluorescenceFileWriter]) -> None:
        self.stxmAnalyzer = STXMAnalyzer(productRepository)
        self.stxmVisualizationEngine = VisualizationEngine(isComplex=False)
        self.probePropagator = ProbePropagator(productRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=True)
        self.exposureAnalyzer = ExposureAnalyzer(productRepository)
        self.exposureVisualizationEngine = VisualizationEngine(isComplex=False)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)
        self.fluorescenceEnhancer = FluorescenceEnhancer(productRepository,
                                                         upscalingStrategyChooser,
                                                         deconvolutionStrategyChooser,
                                                         fluorescenceFileReaderChooser,
                                                         fluorescenceFileWriterChooser)
        self.fluorescenceVisualizationEngine = VisualizationEngine(isComplex=False)
        self.xmcdAnalyzer = XMCDAnalyzer(objectRepository)
        self.xmcdVisualizationEngine = VisualizationEngine(isComplex=False)
