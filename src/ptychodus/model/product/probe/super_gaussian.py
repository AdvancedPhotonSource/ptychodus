from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder
from .settings import ProbeSettings


class SuperGaussianProbeBuilder(ProbeBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'super_gaussian')
        self._settings = settings

        self.annularRadiusInMeters = settings.superGaussianAnnularRadiusInMeters.copy()
        self._add_parameter('annular_radius_m', self.annularRadiusInMeters)

        self.fwhmInMeters = settings.superGaussianWidthInMeters.copy()
        self._add_parameter('full_width_at_half_maximum_m', self.fwhmInMeters)

        self.orderParameter = settings.superGaussianOrderParameter.copy()
        self._add_parameter('order_parameter', self.orderParameter)

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.get_probe_geometry()
        coords = self.get_transverse_coordinates(geometry)

        Z = (
            coords.position_r_m - self.annularRadiusInMeters.get_value()
        ) / self.fwhmInMeters.get_value()
        ZP = numpy.power(2 * Z, 2 * self.orderParameter.get_value())

        return Probe(
            array=self.normalize(numpy.exp(-numpy.log(2) * ZP) + 0j),
            pixel_geometry=geometry.get_pixel_geometry(),
        )
