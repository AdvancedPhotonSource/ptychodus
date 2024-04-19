from ..product import ObjectRepository, ProductRepository
from ..visualization import VisualizationEngine
from .dichroic import DichroicAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator


class AnalysisCore:

    def __init__(self, productRepository: ProductRepository,
                 objectRepository: ObjectRepository) -> None:
        self.probePropagator = ProbePropagator(productRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=True)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)
        self.dichroicAnalyzer = DichroicAnalyzer(objectRepository)
        self.dichroicVisualizationEngine = VisualizationEngine(isComplex=False)
