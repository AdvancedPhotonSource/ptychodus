from __future__ import annotations

import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder


class LissajousScanBuilder(ScanBuilder):

    def __init__(self) -> None:
        super().__init__('lissajous')
        self.numberOfPoints = self._registerIntegerParameter('number_of_points', 100, minimum=0)
        self.amplitudeXInMeters = self._registerRealParameter(
            'amplitude_x_m',
            4.5e-6,
            minimum=0.,
        )
        self.amplitudeYInMeters = self._registerRealParameter(
            'amplitude_y_m',
            4.5e-6,
            minimum=0.,
        )
        self.angularStepXInTurns = self._registerRealParameter('angular_step_x_tr', 0.03)
        self.angularStepYInTurns = self._registerRealParameter('angular_step_y_tr', 0.04)
        self.angularShiftInTurns = self._registerRealParameter('angular_shift_tr', 0.25)

    def copy(self) -> LissajousScanBuilder:
        builder = LissajousScanBuilder()
        builder.numberOfPoints.setValue(self.numberOfPoints.getValue())
        builder.amplitudeXInMeters.setValue(self.amplitudeXInMeters.getValue())
        builder.amplitudeYInMeters.setValue(self.amplitudeYInMeters.getValue())
        builder.angularStepXInTurns.setValue(self.angularStepXInTurns.getValue())
        builder.angularStepYInTurns.setValue(self.angularStepYInTurns.getValue())
        builder.angularShiftInTurns.setValue(self.angularShiftInTurns.getValue())
        return builder

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self.numberOfPoints.getValue()):
            twoPi = 2 * numpy.pi
            thetaX = twoPi * self.angularStepXInTurns.getValue() * index \
                    + self.angularShiftInTurns.getValue()
            thetaY = twoPi * self.angularStepYInTurns.getValue() * index

            point = ScanPoint(
                index=index,
                positionXInMeters=self.amplitudeXInMeters.getValue() * numpy.sin(thetaX),
                positionYInMeters=self.amplitudeYInMeters.getValue() * numpy.sin(thetaY),
            )
            pointList.append(point)

        return Scan(pointList)