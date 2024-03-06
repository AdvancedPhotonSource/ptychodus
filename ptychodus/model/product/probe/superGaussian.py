import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder


class SuperGaussianProbeBuilder(ProbeBuilder):

    def __init__(self, geometryProvider: ProbeGeometryProvider) -> None:
        super().__init__('Super Gaussian')
        self._geometryProvider = geometryProvider

        self.annularRadiusInMeters = self._registerRealParameter(
            'AnnularRadiusInMeters',
            0.,
            minimum=0.,
        )
        self.fwhmInMeters = self._registerRealParameter(
            'FullWidthAtHalfMaximumInMeters',
            1.e-6,
            minimum=0.,
        )
        self.orderParameter = self._registerRealParameter(
            'OrderParameter',
            1.,
            minimum=1.,
        )

    def build(self) -> Probe:
        geometry = self._geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        Z = (coords.positionRInMeters - self.annularRadiusInMeters.getValue()) \
                / self.fwhmInMeters.getValue()
        ZP = numpy.power(2 * Z, 2 * self.orderParameter.getValue())

        return Probe(
            array=self.normalize(numpy.exp(-numpy.log(2) * ZP) + 0j),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
