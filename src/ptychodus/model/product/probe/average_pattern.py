from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from ...diffraction import DiffractionAPI
from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeSequenceBuilder):
    def __init__(
        self,
        settings: ProbeSettings,
        diffraction_api: DiffractionAPI,
    ) -> None:
        super().__init__(settings, 'average_pattern')
        self._settings = settings
        self._diffraction_api = diffraction_api

    def copy(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(self._settings, self._diffraction_api)

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        geometry = geometry_provider.get_probe_geometry()
        assembled_data = self._diffraction_api.get_assembled_data()
        detector_intensity = numpy.mean(assembled_data.get_patterns(), axis=0)

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

        return ProbeSequence(
            array=self.normalize(array),
            opr_weights=None,
            pixel_geometry=geometry.get_pixel_geometry(),
        )
