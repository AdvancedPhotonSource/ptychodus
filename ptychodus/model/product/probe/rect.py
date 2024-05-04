from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder
from .settings import ProbeSettings


class RectangularProbeBuilder(ProbeBuilder):

    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__('rectangular')
        self._settings = settings

        self.widthInMeters = self._registerRealParameter(
            'width_m',
            float(settings.rectangleWidthInMeters.value),
            minimum=0.,
        )
        self.heightInMeters = self._registerRealParameter(
            'height_m',
            float(settings.rectangleHeightInMeters.value),
            minimum=0.,
        )

    def copy(self) -> RectangularProbeBuilder:
        builder = RectangularProbeBuilder(self._settings)
        builder.widthInMeters.setValue(self.widthInMeters.getValue())
        builder.heightInMeters.setValue(self.heightInMeters.getValue())
        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        aX_m = numpy.fabs(coords.positionXInMeters)
        rx_m = self.widthInMeters.getValue() / 2.
        aY_m = numpy.fabs(coords.positionYInMeters)
        ry_m = self.heightInMeters.getValue() / 2.

        isInside = numpy.logical_and(aX_m < rx_m, aY_m < ry_m)

        return Probe(
            array=self.normalize(numpy.where(isInside, 1 + 0j, 0j)),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
