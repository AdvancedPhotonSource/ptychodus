from __future__ import annotations

import numpy

from ptychodus.api.parametric import RealParameter
from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder
from .settings import ProbeSettings


class SuperGaussianProbeBuilder(ProbeBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__("super_gaussian")
        self._settings = settings

        self.annularRadiusInMeters = RealParameter(
            self,
            "annular_radius_m",
            float(settings.superGaussianAnnularRadiusInMeters.getValue()),
            minimum=0.0,
        )
        self.fwhmInMeters = RealParameter(
            self,
            "full_width_at_half_maximum_m",
            float(settings.superGaussianWidthInMeters.getValue()),
            minimum=0.0,
        )
        self.orderParameter = RealParameter(
            self,
            "order_parameter",
            float(settings.superGaussianOrderParameter.getValue()),
            minimum=1.0,
        )

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder(self._settings)
        builder.annularRadiusInMeters.setValue(self.annularRadiusInMeters.getValue())
        builder.fwhmInMeters.setValue(self.fwhmInMeters.getValue())
        builder.orderParameter.setValue(self.orderParameter.getValue())
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
