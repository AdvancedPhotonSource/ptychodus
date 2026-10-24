from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeEnsemble, ProbeGeometryProvider
from ptychodus.api.simulate.probe import generate_rectangular_probe

from .builder import ProbeEnsembleBuilder
from .settings import ProbeSettings


class RectangularProbeBuilder(ProbeEnsembleBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(rng, settings, 'rectangular')
        self._settings = settings

        self.width_m = settings.rectangle_width_m.copy()
        self._add_parameter('width_m', self.width_m)

        self.height_m = settings.rectangle_height_m.copy()
        self._add_parameter('height_m', self.height_m)

        # from the focus to the sample, setting the curvature of the beam at the aperture
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

        # from sample to the focal plane
        self.propagation_distance_m = settings.propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)

    def copy(self) -> RectangularProbeBuilder:
        builder = RectangularProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        probe = generate_rectangular_probe(
            geometry_provider.get_probe_geometry(),
            width_m=self.width_m.get_value(),
            height_m=self.height_m.get_value(),
        )
        probe = self._shape_parametric_field(
            probe,
            geometry_provider,
            defocus_distance_m=self.defocus_distance_m.get_value(),
            propagation_distance_m=self.propagation_distance_m.get_value(),
        )
        return self._rescale_to_photon_count(probe, geometry_provider)
