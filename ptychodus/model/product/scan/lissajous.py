import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder


class LissajousScanBuilder(ScanBuilder):

    def __init__(self) -> None:
        super().__init__('Lissajous')
        self.numberOfPoints = self._registerIntegerParameter('NumberOfPoints', 100, minimum=0)
        self.amplitudeXInMeters = self._registerRealParameter(
            'amplitudeXInMeters',
            4.5e-6,
            minimum=0.,
        )
        self.amplitudeYInMeters = self._registerRealParameter(
            'amplitudeYInMeters',
            4.5e-6,
            minimum=0.,
        )
        self.angularStepXInTurns = self._registerRealParameter('angularStepXInTurns', 0.03)
        self.angularStepYInTurns = self._registerRealParameter('angularStepYInTurns', 0.04)
        self.angularShiftInTurns = self._registerRealParameter('angularShiftInTurns', 0.25)

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
