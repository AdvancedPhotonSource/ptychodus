from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import generate_average_pattern_probe

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
        super().__init__(rng, settings, 'average_pattern')
        self._settings = settings
        self._dataset = dataset

    def copy(self) -> AveragePatternProbeBuilder:
        builder = AveragePatternProbeBuilder(self._rng, self._settings, self._dataset)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        return self._rescale_to_photon_count(
            generate_average_pattern_probe(
                geometry_provider.get_probe_geometry(),
                self._dataset.get_assembled_data(),
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                detector_distance_m=geometry_provider.detector_distance_m,
            ),
            geometry_provider,
        )
