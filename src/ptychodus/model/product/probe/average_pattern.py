from __future__ import annotations

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import generate_average_pattern_probe, rescale_probe_intensity

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
        probe = rescale_probe_intensity(
            generate_average_pattern_probe(
                geometry_provider.get_probe_geometry(),
                self._diffraction_api.get_assembled_data(),
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                detector_distance_m=geometry_provider.detector_distance_m,
            ),
            geometry_provider.probe_photon_count,
        )
        return ProbeSequence(
            array=probe.get_array(),
            opr_weights=None,
            pixel_geometry=probe.get_pixel_geometry(),
        )
