from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from ...patterns import AssembledDiffractionDataset
from .builder import ProbeBuilder
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeBuilder):
    def __init__(
        self,
        settings: ProbeSettings,
        dataset: AssembledDiffractionDataset,
    ) -> None:
        super().__init__(settings, 'average_pattern')
        self._settings = settings
        self._dataset = dataset

    def copy(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(self._settings, self._dataset)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.get_probe_geometry()
        detectorIntensity = numpy.average(self._dataset.get_assembled_patterns(), axis=0)

        pixelGeometry = geometryProvider.get_detector_pixel_geometry()
        propagatorParameters = PropagatorParameters(
            wavelength_m=geometryProvider.probe_wavelength_m,
            width_px=detectorIntensity.shape[-1],
            height_px=detectorIntensity.shape[-2],
            pixel_width_m=pixelGeometry.width_m,
            pixel_height_m=pixelGeometry.height_m,
            propagation_distance_m=-geometryProvider.detector_distance_m,
        )
        propagator = FresnelTransformPropagator(propagatorParameters)
        array = propagator.propagate(numpy.sqrt(detectorIntensity).astype(complex))

        return Probe(
            array=self.normalize(array),
            pixel_geometry=geometry.get_pixel_geometry(),
        )
