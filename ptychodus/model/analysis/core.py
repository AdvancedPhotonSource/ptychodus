from ..product import ObjectRepository, ProbeRepository
from ..visualization import VisualizationEngine
from .dichroic import DichroicAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator


class AnalysisCore:

    def __init__(self, probeRepository: ProbeRepository,
                 objectRepository: ObjectRepository) -> None:
        self.probePropagator = ProbePropagator(probeRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=True)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)
        self.dichroicAnalyzer = DichroicAnalyzer(objectRepository)
        self.dichroicVisualizationEngine = VisualizationEngine(isComplex=False)
