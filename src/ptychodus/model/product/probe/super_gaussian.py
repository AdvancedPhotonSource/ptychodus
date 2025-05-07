from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class SuperGaussianProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'super_gaussian')
        self._settings = settings

        self.annular_radius_m = settings.super_gaussian_annular_radius_m.copy()
        self._add_parameter('annular_radius_m', self.annular_radius_m)

        self.fwhm_m = settings.super_gaussian_width_m.copy()
        self._add_parameter('full_width_at_half_maximum_m', self.fwhm_m)

        self.order_parameter = settings.super_gaussian_order_parameter.copy()
        self._add_parameter('order_parameter', self.order_parameter)

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        geometry = geometry_provider.get_probe_geometry()
        coords = self.get_transverse_coordinates(geometry)

        Z = (  # noqa: N806
            coords.position_r_m - self.annular_radius_m.get_value()
        ) / self.fwhm_m.get_value()
        ZP = numpy.power(2 * Z, 2 * self.order_parameter.get_value())  # noqa: N806

        return ProbeSequence(
            array=self.normalize(numpy.exp(-numpy.log(2) * ZP) + 0j),
            opr_weights=None,
            pixel_geometry=geometry.get_pixel_geometry(),
        )
