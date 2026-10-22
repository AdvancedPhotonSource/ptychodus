from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeEnsemble, ProbeGeometryProvider
from ptychodus.api.simulate.probe import generate_average_pattern_probe, propagate_probe

from ...diffraction import AssembledDiffractionDataset
from .builder import ProbeEnsembleBuilder
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeEnsembleBuilder):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        dataset: AssembledDiffractionDataset,
    ) -> None:
        super().__init__(rng, settings, 'average_pattern')
        self._settings = settings
        self._dataset = dataset
        self.propagation_distance_m = settings.propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)

    def copy(self) -> AveragePatternProbeBuilder:
        builder = AveragePatternProbeBuilder(self._rng, self._settings, self._dataset)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        probe = generate_average_pattern_probe(
            geometry_provider.get_probe_geometry(),
            self._dataset.get_assembled_data(),
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            detector_distance_m=geometry_provider.detector_distance_m,
            far_field=geometry_provider.far_field,
        )
        distance_m = self.propagation_distance_m.get_value()

        # Back-propagation lands the probe at the object plane the declared geometry
        # describes, so the plane is only as good as that geometry. Propagating from
        # here refines where the probe sits without moving the object pixel size,
        # which editing the geometry would. Skipped at zero: the angular spectrum is
        # the identity there, but only after a full transform pair.
        if distance_m != 0.0:
            probe = propagate_probe(
                probe,
                photon_wavelength_m=geometry_provider.photon_wavelength_m,
                propagation_distance_m=distance_m,
            )

        return self._rescale_to_photon_count(probe, geometry_provider)
