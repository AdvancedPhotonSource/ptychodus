from __future__ import annotations

import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class ConcentricScanBuilder(ScanBuilder):
    """https://doi.org/10.1088/1367-2630/12/3/035017"""

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'concentric')
        self._settings = settings

        self.radialStepSizeInMeters = settings.radialStepSizeInMeters.copy()
        self._addParameter('radial_step_size_m', self.radialStepSizeInMeters)

        self.numberOfShells = settings.numberOfShells.copy()
        self._addParameter('number_of_shells', self.numberOfShells)

        self.numberOfPointsInFirstShell = settings.numberOfPointsInFirstShell.copy()
        self._addParameter('number_of_points_1st_shell', self.numberOfPointsInFirstShell)

    def copy(self) -> ConcentricScanBuilder:
        builder = ConcentricScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value.getValue())

        return builder

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
