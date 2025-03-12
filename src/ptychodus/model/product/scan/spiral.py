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
        self.numberOfPoints.set_value(
            settings.numberOfPointsX.get_value() * settings.numberOfPointsY.get_value()
        )
        self._add_parameter('number_of_points', self.numberOfPoints)

        self._numberOfPoints = settings.numberOfPointsY.copy()
        self._numberOfPoints.set_value(1)
        self._add_parameter('_number_of_points', self._numberOfPoints)

        self.radiusScalarInMeters = settings.radiusScalarInMeters.copy()
        self._add_parameter('radius_scalar_m', self.radiusScalarInMeters)

    def copy(self) -> SpiralScanBuilder:
        builder = SpiralScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self.numberOfPoints.get_value()):
            radiusInMeters = self.radiusScalarInMeters.get_value() * numpy.sqrt(index)
            divergenceAngleInRadians = (3.0 - numpy.sqrt(5)) * numpy.pi
            thetaInRadians = divergenceAngleInRadians * index

            point = ScanPoint(
                index=index,
                position_x_m=radiusInMeters * numpy.cos(thetaInRadians),
                position_y_m=radiusInMeters * numpy.sin(thetaInRadians),
            )
            pointList.append(point)

        return Scan(pointList)
