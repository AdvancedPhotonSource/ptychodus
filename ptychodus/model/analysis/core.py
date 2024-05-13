from ..product import ObjectRepository, ProductRepository
from ..visualization import VisualizationEngine
from .exposure import ExposureAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator
from .stxm import STXMAnalyzer
from .xmcd import XMCDAnalyzer


class AnalysisCore:

    def __init__(self, productRepository: ProductRepository,
                 objectRepository: ObjectRepository) -> None:
        self.stxmAnalyzer = STXMAnalyzer(productRepository)
        self.stxmVisualizationEngine = VisualizationEngine(isComplex=False)
        self.probePropagator = ProbePropagator(productRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=True)
        self.exposureAnalyzer = ExposureAnalyzer(productRepository)
        self.exposureVisualizationEngine = VisualizationEngine(isComplex=False)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)
        self.xmcdAnalyzer = XMCDAnalyzer(objectRepository)
        self.xmcdVisualizationEngine = VisualizationEngine(isComplex=False)
