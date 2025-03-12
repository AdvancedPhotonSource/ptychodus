from __future__ import annotations

import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class LissajousScanBuilder(ScanBuilder):
    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'lissajous')
        self._settings = settings

        self.numberOfPoints = settings.numberOfPointsX.copy()
        self.numberOfPoints.set_value(
            settings.numberOfPointsX.get_value() * settings.numberOfPointsY.get_value()
        )
        self._add_parameter('number_of_points', self.numberOfPoints)

        self._numberOfPoints = settings.numberOfPointsY.copy()
        self._numberOfPoints.set_value(1)
        self._add_parameter('_number_of_points', self._numberOfPoints)

        self.amplitudeXInMeters = settings.amplitudeXInMeters.copy()
        self._add_parameter('amplitude_x_m', self.amplitudeXInMeters)

        self.amplitudeYInMeters = settings.amplitudeYInMeters.copy()
        self._add_parameter('amplitude_y_m', self.amplitudeYInMeters)

        self.angularStepXInTurns = settings.angularStepXInTurns.copy()
        self._add_parameter('angular_step_x_tr', self.angularStepXInTurns)

        self.angularStepYInTurns = settings.angularStepYInTurns.copy()
        self._add_parameter('angular_step_y_tr', self.angularStepYInTurns)

        self.angularShiftInTurns = settings.angularShiftInTurns.copy()
        self._add_parameter('angular_shift_tr', self.angularShiftInTurns)

    def copy(self) -> LissajousScanBuilder:
        builder = LissajousScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self.numberOfPoints.get_value()):
            twoPi = 2 * numpy.pi
            thetaX = (
                twoPi * self.angularStepXInTurns.get_value() * index
                + self.angularShiftInTurns.get_value()
            )
            thetaY = twoPi * self.angularStepYInTurns.get_value() * index

            point = ScanPoint(
                index=index,
                position_x_m=self.amplitudeXInMeters.get_value() * numpy.sin(thetaX),
                position_y_m=self.amplitudeYInMeters.get_value() * numpy.sin(thetaY),
            )
            pointList.append(point)

        return Scan(pointList)
