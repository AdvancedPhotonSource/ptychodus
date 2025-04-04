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

    def build(self, geometry_provider: ProbeGeometryProvider) -> Probe:
        geometry = geometry_provider.get_probe_geometry()
        detector_intensity = numpy.average(self._dataset.get_assembled_patterns(), axis=0)

        pixel_geometry = geometry_provider.get_detector_pixel_geometry()
        propagator_parameters = PropagatorParameters(
            wavelength_m=geometry_provider.probe_wavelength_m,
            width_px=detector_intensity.shape[-1],
            height_px=detector_intensity.shape[-2],
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            propagation_distance_m=-geometry_provider.detector_distance_m,
        )
        propagator = FresnelTransformPropagator(propagator_parameters)
        array = propagator.propagate(numpy.sqrt(detector_intensity).astype(complex))

        return Probe(
            array=self.normalize(array),
            pixel_geometry=geometry.get_pixel_geometry(),
        )
