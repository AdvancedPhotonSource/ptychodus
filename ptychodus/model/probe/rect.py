import numpy

from ...api.probe import Probe, ProbeGeometryProvider
from .builder import ProbeBuilder


class RectangularProbeBuilder(ProbeBuilder):

    def __init__(self, geometryProvider: ProbeGeometryProvider) -> None:
        super().__init__('Rectangular')
        self._geometryProvider = geometryProvider

        self.widthInMeters = self._registerRealParameter(
            'WidthInMeters',
            1.e-6,
            minimum=0.,
        )
        self.heightInMeters = self._registerRealParameter(
            'HeightInMeters',
            1.e-6,
            minimum=0.,
        )

    def build(self) -> Probe:
        geometry = self._geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        aX_m = numpy.fabs(coords.positionXInMeters)
        rx_m = self.widthInMeters.getValue() / 2.
        aY_m = numpy.fabs(coords.positionYInMeters)
        ry_m = self.heightInMeters.getValue() / 2.

        return Probe(
            array=self.normalize(numpy.where(aX_m < rx_m and aY_m < ry_m, 1., 0.) + 0j),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
