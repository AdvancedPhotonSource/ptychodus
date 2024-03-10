from ...api.plugins import PluginChooser
from ...api.visualize import ScalarTransformation
from ..image import ImageCore
from ..product import ObjectRepository, ProbeRepository
from .dichroic import DichroicAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator


class AnalysisCore:

    def __init__(self, scalarTransformations: PluginChooser[ScalarTransformation],
                 probeRepository: ProbeRepository, objectRepository: ObjectRepository) -> None:
        self.probePropagator = ProbePropagator(probeRepository)
        self.probePropagatorImageCore = ImageCore(scalarTransformations, isComplex=True)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)
        self.dichroicAnalyzer = DichroicAnalyzer(objectRepository)
        self.dichroicImageCore = ImageCore(scalarTransformations, isComplex=False)
