from ..product import ObjectRepository, ProbeRepository
from .dichroic import DichroicAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator


class AnalysisCore:

    def __init__(self, probeRepository: ProbeRepository,
                 objectRepository: ObjectRepository) -> None:
        self.probePropagator = ProbePropagator(probeRepository)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)
        self.dichroicAnalyzer = DichroicAnalyzer(objectRepository)
