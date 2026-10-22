from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeEnsemble, ProbeGeometryProvider
from ptychodus.api.simulate.probe import propagate_probe, generate_disk_probe

from .builder import ProbeEnsembleBuilder
from .settings import ProbeSettings


class DiskProbeBuilder(ProbeEnsembleBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(rng, settings, 'disk')
        self._settings = settings

        self.diameter_m = settings.disk_diameter_m.copy()
        self._add_parameter('diameter_m', self.diameter_m)

        # from sample to the focal plane
        self.propagation_distance_m = settings.propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)

    def copy(self) -> DiskProbeBuilder:
        builder = DiskProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        return self._rescale_to_photon_count(
            propagate_probe(
                generate_disk_probe(
                    geometry_provider.get_probe_geometry(),
                    radius_m=self.diameter_m.get_value() / 2.0,
                ),
                photon_wavelength_m=geometry_provider.photon_wavelength_m,
                propagation_distance_m=self.propagation_distance_m.get_value(),
            ),
            geometry_provider,
        )
