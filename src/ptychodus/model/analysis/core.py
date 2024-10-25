import logging

from ptychodus.api.settings import SettingsRegistry

from ..product import ObjectRepository, ProductRepository
from ..reconstructor import DiffractionPatternPositionMatcher
from ..visualization import VisualizationEngine
from .exposure import ExposureAnalyzer
from .frc import FourierRingCorrelator
from .propagator import ProbePropagator
from .settings import ProbePropagationSettings
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
    ) -> None:
        self.stxmSimulator = STXMSimulator(dataMatcher)
        self.stxmVisualizationEngine = VisualizationEngine(isComplex=False)

        self._probePropagationSettings = ProbePropagationSettings(settingsRegistry)
        self.probePropagator = ProbePropagator(self._probePropagationSettings, productRepository)
        self.probePropagatorVisualizationEngine = VisualizationEngine(isComplex=False)
        self.exposureAnalyzer = ExposureAnalyzer(productRepository)
        self.exposureVisualizationEngine = VisualizationEngine(isComplex=False)
        self.fourierRingCorrelator = FourierRingCorrelator(objectRepository)

        self.xmcdAnalyzer = XMCDAnalyzer(objectRepository)
        self.xmcdVisualizationEngine = VisualizationEngine(isComplex=False)
