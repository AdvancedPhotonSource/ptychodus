from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import defocus_probe, generate_disk_probe, rescale_probe_intensity

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class DiskProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(settings, 'disk')
        self._rng = rng
        self._settings = settings

        self.diameter_m = settings.disk_diameter_m.copy()
        self._add_parameter('diameter_m', self.diameter_m)

        # from sample to the focal plane
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

    def copy(self) -> DiskProbeBuilder:
        builder = DiskProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe = rescale_probe_intensity(
            defocus_probe(
                generate_disk_probe(
                    geometry_provider.get_probe_geometry(),
                    radius_m=self.diameter_m.get_value() / 2.0,
                ),
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                defocus_distance_m=self.defocus_distance_m.get_value(),
            ),
            geometry_provider.probe_photon_count,
        )
        return self._build_probe_modes(self._rng, probe, geometry_provider.num_scan_points)
