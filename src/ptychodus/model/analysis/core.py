import logging

from ptychodus.api.settings import SettingsRegistry

from ..product import ObjectRepository, ProductRepository
from ..reconstructor import DiffractionPatternPositionMatcher
from ..visualization import VisualizationEngine
from .fourier import FourierAnalyzer
from .frc import FourierRingCorrelator
from .illumination import IlluminationMapper
from .propagator import ProbePropagator
from .settings import ProbePropagationSettings
from .stxm import STXMSimulator
from .xmcd import XMCDAnalyzer

logger = logging.getLogger(__name__)


class AnalysisCore:
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        data_matcher: DiffractionPatternPositionMatcher,
        product_repository: ProductRepository,
        object_repository: ObjectRepository,
    ) -> None:
        self.stxm_simulator = STXMSimulator(data_matcher)
        self.stxm_visualization_engine = VisualizationEngine(is_complex=False)

        self._probe_propagation_settings = ProbePropagationSettings(settings_registry)
        self.probe_propagator = ProbePropagator(
            self._probe_propagation_settings, product_repository
        )
        self.probe_propagator_visualization_engine = VisualizationEngine(is_complex=False)

        self.exposure_analyzer = IlluminationMapper(product_repository)
        self.exposure_visualization_engine = VisualizationEngine(is_complex=False)

        self.fourier_ring_correlator = FourierRingCorrelator(object_repository)

        self.fourier_analyzer = FourierAnalyzer(product_repository)
        self.fourier_real_space_visualization_engine = VisualizationEngine(is_complex=True)
        self.fourier_reciprocal_space_visualization_engine = VisualizationEngine(is_complex=True)

        self.xmcd_analyzer = XMCDAnalyzer(product_repository)
        self.xmcd_visualization_engine = VisualizationEngine(is_complex=False)
