from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import generate_average_pattern_probe, rescale_probe_intensity

from ...diffraction import AssembledDiffractionDataset
from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeSequenceBuilder):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        dataset: AssembledDiffractionDataset,
    ) -> None:
        super().__init__(settings, 'average_pattern')
        self._rng = rng
        self._settings = settings
        self._dataset = dataset

    def copy(self) -> AveragePatternProbeBuilder:
        builder = AveragePatternProbeBuilder(self._rng, self._settings, self._dataset)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe = rescale_probe_intensity(
            generate_average_pattern_probe(
                geometry_provider.get_probe_geometry(),
                self._dataset.get_assembled_data(),
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                detector_distance_m=geometry_provider.detector_distance_m,
            ),
            geometry_provider.probe_photon_count,
        )
        return self._build_probe_modes(self._rng, probe, geometry_provider.num_scan_points)
