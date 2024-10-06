from __future__ import annotations

import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class SpiralScanBuilder(ScanBuilder):
    """https://doi.org/10.1364/OE.22.012634"""

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'spiral')
        self._settings = settings

        self.numberOfPoints = settings.numberOfPointsX.copy()
        self.numberOfPoints.setValue(
            settings.numberOfPointsX.getValue() * settings.numberOfPointsY.getValue()
        )
        self._addParameter('number_of_points', self.numberOfPoints)

        self._numberOfPoints = settings.numberOfPointsY.copy()
        self._numberOfPoints.setValue(1)
        self._addParameter('_number_of_points', self._numberOfPoints)

        self.radiusScalarInMeters = settings.radiusScalarInMeters.copy()
        self._addParameter('radius_scalar_m', self.radiusScalarInMeters)

    def copy(self) -> SpiralScanBuilder:
        builder = SpiralScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value)

        return builder

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self.numberOfPoints.getValue()):
            radiusInMeters = self.radiusScalarInMeters.getValue() * numpy.sqrt(index)
            divergenceAngleInRadians = (3.0 - numpy.sqrt(5)) * numpy.pi
            thetaInRadians = divergenceAngleInRadians * index

            point = ScanPoint(
                index=index,
                positionXInMeters=radiusInMeters * numpy.cos(thetaInRadians),
                positionYInMeters=radiusInMeters * numpy.sin(thetaInRadians),
            )
            pointList.append(point)

        return Scan(pointList)
