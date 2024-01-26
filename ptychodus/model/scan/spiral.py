import numpy

from ...api.scan import Scan, ScanPoint
from .builder import ScanBuilder


class SpiralScanBuilder(ScanBuilder):
    '''https://doi.org/10.1364/OE.22.012634'''

    def __init__(self) -> None:
        super().__init__('Spiral')
        self.numberOfPoints = self._registerIntegerParameter('NumberOfPoints', 100, minimum=0)
        self.radiusScalarInMeters = self._registerRealParameter(
            'RadiusScalarInMeters',
            5e-7,
            minimum=0.,
        )

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self.numberOfPoints.getValue()):
            radiusInMeters = self.radiusScalarInMeters.getValue() * numpy.sqrt(index)
            divergenceAngleInRadians = (3. - numpy.sqrt(5)) * numpy.pi
            thetaInRadians = divergenceAngleInRadians * index

            point = ScanPoint(
                index=index,
                positionXInMeters=radiusInMeters * numpy.cos(thetaInRadians),
                positionYInMeters=radiusInMeters * numpy.sin(thetaInRadians),
            )
            pointList.append(point)

        return Scan(pointList)
