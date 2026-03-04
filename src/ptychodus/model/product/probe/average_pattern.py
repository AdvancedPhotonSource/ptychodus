from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import generate_average_pattern_probe, rescale_probe_intensity

from ...diffraction import DiffractionAPI
from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeSequenceBuilder):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        diffraction_api: DiffractionAPI,
    ) -> None:
        super().__init__(settings, 'average_pattern')
        self._rng = rng
        self._settings = settings
        self._diffraction_api = diffraction_api

    def copy(self) -> AveragePatternProbeBuilder:
        builder = AveragePatternProbeBuilder(self._rng, self._settings, self._diffraction_api)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe = rescale_probe_intensity(
            generate_average_pattern_probe(
                geometry_provider.get_probe_geometry(),
                self._diffraction_api.get_assembled_data(),
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                detector_distance_m=geometry_provider.detector_distance_m,
            ),
            geometry_provider.probe_photon_count,
        )
        return self._build_probe_modes(self._rng, probe, geometry_provider.num_scan_points)
