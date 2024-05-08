from ..product import ObjectRepository, ProductRepository
from ..visualization import VisualizationEngine
from .dichroic import DichroicAnalyzer
from .exposure import ExposureAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator
from .stxm import STXMAnalyzer


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
        self.dichroicAnalyzer = DichroicAnalyzer(objectRepository)
        self.dichroicVisualizationEngine = VisualizationEngine(isComplex=False)
