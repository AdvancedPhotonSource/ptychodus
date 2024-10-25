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
        self.numberOfPoints.setValue(
            settings.numberOfPointsX.getValue() * settings.numberOfPointsY.getValue()
        )
        self._addParameter('number_of_points', self.numberOfPoints)

        self._numberOfPoints = settings.numberOfPointsY.copy()
        self._numberOfPoints.setValue(1)
        self._addParameter('_number_of_points', self._numberOfPoints)

        self.amplitudeXInMeters = settings.amplitudeXInMeters.copy()
        self._addParameter('amplitude_x_m', self.amplitudeXInMeters)

        self.amplitudeYInMeters = settings.amplitudeYInMeters.copy()
        self._addParameter('amplitude_y_m', self.amplitudeYInMeters)

        self.angularStepXInTurns = settings.angularStepXInTurns.copy()
        self._addParameter('angular_step_x_tr', self.angularStepXInTurns)

        self.angularStepYInTurns = settings.angularStepYInTurns.copy()
        self._addParameter('angular_step_y_tr', self.angularStepYInTurns)

        self.angularShiftInTurns = settings.angularShiftInTurns.copy()
        self._addParameter('angular_shift_tr', self.angularShiftInTurns)

    def copy(self) -> LissajousScanBuilder:
        builder = LissajousScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value.getValue())

        return builder

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self.numberOfPoints.getValue()):
            twoPi = 2 * numpy.pi
            thetaX = (
                twoPi * self.angularStepXInTurns.getValue() * index
                + self.angularShiftInTurns.getValue()
            )
            thetaY = twoPi * self.angularStepYInTurns.getValue() * index

            point = ScanPoint(
                index=index,
                positionXInMeters=self.amplitudeXInMeters.getValue() * numpy.sin(thetaX),
                positionYInMeters=self.amplitudeYInMeters.getValue() * numpy.sin(thetaY),
            )
            pointList.append(point)

        return Scan(pointList)
