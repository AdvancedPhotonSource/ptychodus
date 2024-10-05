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
        self._addParameter('annular_radius_m', self.annularRadiusInMeters)

        self.fwhmInMeters = settings.superGaussianWidthInMeters.copy()
        self._addParameter('full_width_at_half_maximum_m', self.fwhmInMeters)

        self.orderParameter = settings.superGaussianOrderParameter.copy()
        self._addParameter('order_parameter', self.orderParameter)

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value)

        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        Z = (
            coords.positionRInMeters - self.annularRadiusInMeters.getValue()
        ) / self.fwhmInMeters.getValue()
        ZP = numpy.power(2 * Z, 2 * self.orderParameter.getValue())

        return Probe(
            array=self.normalize(numpy.exp(-numpy.log(2) * ZP) + 0j),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
