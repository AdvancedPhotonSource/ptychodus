from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import (
    defocus_probe,
    generate_rectangular_probe,
    rescale_probe_intensity,
)

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class RectangularProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(settings, 'rectangular')
        self._rng = rng
        self._settings = settings

        self.width_m = settings.rectangle_width_m.copy()
        self._add_parameter('width_m', self.width_m)

        self.height_m = settings.rectangle_height_m.copy()
        self._add_parameter('height_m', self.height_m)

        # from sample to the focal plane
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

    def copy(self) -> RectangularProbeBuilder:
        builder = RectangularProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe = rescale_probe_intensity(
            defocus_probe(
                generate_rectangular_probe(
                    geometry_provider.get_probe_geometry(),
                    width_m=self.width_m.get_value(),
                    height_m=self.height_m.get_value(),
                ),
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                defocus_distance_m=self.defocus_distance_m.get_value(),
            ),
            geometry_provider.probe_photon_count,
        )
        return self._build_probe_modes(self._rng, probe, geometry_provider.num_scan_points)
