import numpy

from ptychodus.api.settings import SettingsRegistry

from ..diffraction import DiffractionDatasetRepository
from ..product import ProbePositionsRepository, ProductRepository
from ..visualization import VisualizationEngine
from .affine import AffineTransformEstimator
from .diffraction import DiffractionSimulator
from .fourier import FourierAnalyzer
from .frc import FourierRingCorrelator
from .illumination import IlluminationMapper
from .opr import OPRModeAnalyzer
from .overlap import ProbeOverlapAnalyzer
from .propagator import ProbePropagator
from .residuals import ResidualAnalyzer
from .settings import (
    AffineTransformEstimatorSettings,
    DiffractionSimulatorSettings,
    ProbePropagatorSettings,
)
from .xmcd import XMCDAnalyzer


class AnalysisCore:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings_registry: SettingsRegistry,
        diffraction_repository: DiffractionDatasetRepository,
        product_repository: ProductRepository,
        probe_positions_repository: ProbePositionsRepository,
    ) -> None:
        self._affine_transform_estimator_settings = AffineTransformEstimatorSettings(
            settings_registry
        )
        self.affine_transform_estimator = AffineTransformEstimator(
            rng, self._affine_transform_estimator_settings, probe_positions_repository
        )

        self.diffraction_simulator_settings = DiffractionSimulatorSettings(settings_registry)
        self.diffraction_simulator = DiffractionSimulator(
            rng, self.diffraction_simulator_settings, diffraction_repository, product_repository
        )

        self.fourier_analyzer = FourierAnalyzer(product_repository)
        self.fourier_real_space_visualization_engine = VisualizationEngine(is_complex=True)
        self.fourier_reciprocal_space_visualization_engine = VisualizationEngine(is_complex=True)

        self.fourier_ring_correlator = FourierRingCorrelator(product_repository)

        self.illumination_mapper = IlluminationMapper(product_repository)
        self.illumination_visualization_engine = VisualizationEngine(is_complex=False)

        self.opr_mode_analyzer = OPRModeAnalyzer(product_repository)
        # Complex so the dialog can render the composed mode and the deviation from its
        # across-scan mean as wavefields: OPR moves amplitude and phase together, and the
        # default 'Complex' renderer shows both at once.
        self.opr_mode_visualization_engine = VisualizationEngine(is_complex=True)

        self.probe_overlap_analyzer = ProbeOverlapAnalyzer(product_repository)
        self.probe_overlap_visualization_engine = VisualizationEngine(is_complex=False)

        self.probe_propagator_settings = ProbePropagatorSettings(settings_registry)
        self.probe_propagator = ProbePropagator(self.probe_propagator_settings, product_repository)
        # Complex so the propagation dialog can render a single incoherent mode's
        # wavefield; Real up front because the dialog opens on the mode-summed
        # intensity, for which the is_complex default of 'Complex' paints a flat hue.
        self.probe_propagator_visualization_engine = VisualizationEngine(is_complex=True)
        self.probe_propagator_visualization_engine.set_renderer('Real')

        self.residual_analyzer = ResidualAnalyzer(product_repository)
        self.residual_real_space_visualization_engine = VisualizationEngine(is_complex=False)
        self.residual_reciprocal_space_visualization_engine = VisualizationEngine(is_complex=False)

        self.xmcd_analyzer = XMCDAnalyzer(product_repository)
        self.xmcd_structural_visualization_engine = VisualizationEngine(is_complex=True)
        self.xmcd_magnetic_visualization_engine = VisualizationEngine(is_complex=True)
