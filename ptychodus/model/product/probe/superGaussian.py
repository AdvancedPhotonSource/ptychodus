from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder


class SuperGaussianProbeBuilder(ProbeBuilder):

    def __init__(self) -> None:
        super().__init__('super_gaussian')

        self.annularRadiusInMeters = self._registerRealParameter(
            'annular_radius_m',
            0.,
            minimum=0.,
        )
        self.fwhmInMeters = self._registerRealParameter(
            'full_width_at_half_maximum_m',
            1.e-6,
            minimum=0.,
        )
        self.orderParameter = self._registerRealParameter(
            'order_parameter',
            1.,
            minimum=1.,
        )

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder()
        builder.annularRadiusInMeters.setValue(self.annularRadiusInMeters.getValue())
        builder.fwhmInMeters.setValue(self.fwhmInMeters.getValue())
        builder.orderParameter.setValue(self.orderParameter.getValue())
        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        Z = (coords.positionRInMeters - self.annularRadiusInMeters.getValue()) \
                / self.fwhmInMeters.getValue()
        ZP = numpy.power(2 * Z, 2 * self.orderParameter.getValue())

        return Probe(
            array=self.normalize(numpy.exp(-numpy.log(2) * ZP) + 0j),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
