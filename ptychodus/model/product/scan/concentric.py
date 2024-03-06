import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder


class ConcentricScanBuilder(ScanBuilder):
    '''https://doi.org/10.1088/1367-2630/12/3/035017'''

    def __init__(self) -> None:
        super().__init__('Concentric')
        self.radialStepSizeInMeters = self._registerRealParameter(
            'RadialStepSizeInMeters',
            1e-6,
            minimum=0.,
        )
        self.numberOfShells = self._registerIntegerParameter('NumberOfShells', 5, minimum=0)
        self.numberOfPointsInFirstShell = self._registerIntegerParameter(
            'NumberOfPointsInFirstShell', 10, minimum=0)

    @property
    def _numberOfPoints(self) -> int:
        numberOfShells = self.numberOfShells.getValue()
        triangle = (numberOfShells * (numberOfShells + 1)) // 2
        return triangle * self.numberOfPointsInFirstShell.getValue()

    def build(self) -> Scan:
        pointList: list[ScanPoint] = list()

        for index in range(self._numberOfPoints):
            triangle = index // self.numberOfPointsInFirstShell.getValue()
            shellIndex = int((1 + numpy.sqrt(1 + 8 * triangle)) / 2) - 1  # see OEIS A002024
            shellTriangle = (shellIndex * (shellIndex + 1)) // 2
            firstIndexInShell = self.numberOfPointsInFirstShell.getValue() * shellTriangle
            pointIndexInShell = index - firstIndexInShell

            radiusInMeters = self.radialStepSizeInMeters.getValue() * (shellIndex + 1)
            numberOfPointsInShell = self.numberOfPointsInFirstShell.getValue() * (shellIndex + 1)
            thetaInRadians = 2 * numpy.pi * pointIndexInShell / numberOfPointsInShell

            point = ScanPoint(
                index=index,
                positionXInMeters=radiusInMeters * numpy.cos(thetaInRadians),
                positionYInMeters=radiusInMeters * numpy.sin(thetaInRadians),
            )
            pointList.append(point)

        return Scan(pointList)
