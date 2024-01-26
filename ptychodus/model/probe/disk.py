import numpy

from ...api.probe import Probe, ProbeGeometryProvider
from .builder import ProbeBuilder


class DiskProbeBuilder(ProbeBuilder):

    def __init__(self, geometryProvider: ProbeGeometryProvider) -> None:
        super().__init__('Disk')
        self._geometryProvider = geometryProvider

        self.diameterInMeters = self._registerRealParameter(
            'DiameterInMeters',
            1.e-6,
            minimum=0.,
        )

    def build(self) -> Probe:
        geometry = self._geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        R_m = coords.positionRInMeters
        r_m = self.diameterInMeters.getValue() / 2.

        return Probe(
            array=self.normalize(numpy.where(R_m < r_m, 1., 0.) + 0j),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
