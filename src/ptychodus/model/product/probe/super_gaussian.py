from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.simulate.probe import generate_super_gaussian_probe

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class SuperGaussianProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(rng, settings, 'super_gaussian')
        self._settings = settings

        self.annular_radius_m = settings.super_gaussian_annular_radius_m.copy()
        self._add_parameter('annular_radius_m', self.annular_radius_m)

        self.fwhm_m = settings.super_gaussian_width_m.copy()
        self._add_parameter('full_width_at_half_maximum_m', self.fwhm_m)

        self.order_parameter = settings.super_gaussian_order_parameter.copy()
        self._add_parameter('order_parameter', self.order_parameter)

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        return self._rescale_to_photon_count(
            generate_super_gaussian_probe(
                geometry_provider.get_probe_geometry(),
                annular_radius_m=self.annular_radius_m.get_value(),
                fwhm_m=self.fwhm_m.get_value(),
                order_parameter=self.order_parameter.get_value(),
            ),
            geometry_provider,
        )
