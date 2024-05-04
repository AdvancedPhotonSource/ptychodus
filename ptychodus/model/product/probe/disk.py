from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder
from .settings import ProbeSettings


class DiskProbeBuilder(ProbeBuilder):

    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__('disk')
        self._settings = settings

        self.diameterInMeters = self._registerRealParameter(
            'diameter_m',
            float(settings.diskDiameterInMeters.value),
            minimum=0.,
        )

    def copy(self) -> DiskProbeBuilder:
        builder = DiskProbeBuilder(self._settings)
        builder.diameterInMeters.setValue(self.diameterInMeters.getValue())
        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        R_m = coords.positionRInMeters
        r_m = self.diameterInMeters.getValue() / 2.

        return Probe(
            array=self.normalize(numpy.where(R_m < r_m, 1 + 0j, 0j)),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
