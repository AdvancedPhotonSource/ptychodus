import logging

import numpy

from ptychodus.api.settings import SettingsRegistry

from ..diffraction import AssembledDiffractionDataset
from ..product import ObjectRepository, ProbePositionsRepository, ProductRepository
from ..visualization import VisualizationEngine
from .affine import AffineTransformEstimator
from .diffraction import DiffractionSimulator
from .fourier import FourierAnalyzer
from .frc import FourierRingCorrelator
from .illumination import IlluminationMapper
from .propagator import ProbePropagator
from .settings import AffineTransformEstimatorSettings, ProbePropagationSettings
from .xmcd import XMCDAnalyzer

logger = logging.getLogger(__name__)


class AnalysisCore:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings_registry: SettingsRegistry,
        dataset: AssembledDiffractionDataset,
        product_repository: ProductRepository,
        object_repository: ObjectRepository,
        probe_positions_repository: ProbePositionsRepository,
    ) -> None:
        self._affine_transform_estimator_settings = AffineTransformEstimatorSettings(
            settings_registry
        )
        self.affine_transform_estimator = AffineTransformEstimator(
            rng, self._affine_transform_estimator_settings, probe_positions_repository
        )

        self.diffraction_simulator = DiffractionSimulator(rng, dataset, product_repository)
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
        self.xmcd_structural_visualization_engine = VisualizationEngine(is_complex=True)
        self.xmcd_magnetic_visualization_engine = VisualizationEngine(is_complex=True)
