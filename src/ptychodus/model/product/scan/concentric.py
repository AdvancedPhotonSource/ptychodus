from __future__ import annotations

import numpy

from ptychodus.api.scan import PositionSequence, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class ConcentricScanBuilder(ScanBuilder):
    """https://doi.org/10.1088/1367-2630/12/3/035017"""

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'concentric')
        self._settings = settings

        self.radialStepSizeInMeters = settings.radialStepSizeInMeters.copy()
        self._add_parameter('radial_step_size_m', self.radialStepSizeInMeters)

        self.numberOfShells = settings.numberOfShells.copy()
        self._add_parameter('number_of_shells', self.numberOfShells)

        self.numberOfPointsInFirstShell = settings.numberOfPointsInFirstShell.copy()
        self._add_parameter('number_of_points_1st_shell', self.numberOfPointsInFirstShell)

    def copy(self) -> ConcentricScanBuilder:
        builder = ConcentricScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def _numberOfPoints(self) -> int:
        numberOfShells = self.numberOfShells.get_value()
        triangle = (numberOfShells * (numberOfShells + 1)) // 2
        return triangle * self.numberOfPointsInFirstShell.get_value()

    def build(self) -> PositionSequence:
        pointList: list[ScanPoint] = list()

        for index in range(self._numberOfPoints):
            triangle = index // self.numberOfPointsInFirstShell.get_value()
            shellIndex = int((1 + numpy.sqrt(1 + 8 * triangle)) / 2) - 1  # see OEIS A002024
            shellTriangle = (shellIndex * (shellIndex + 1)) // 2
            firstIndexInShell = self.numberOfPointsInFirstShell.get_value() * shellTriangle
            pointIndexInShell = index - firstIndexInShell

            radiusInMeters = self.radialStepSizeInMeters.get_value() * (shellIndex + 1)
            numberOfPointsInShell = self.numberOfPointsInFirstShell.get_value() * (shellIndex + 1)
            thetaInRadians = 2 * numpy.pi * pointIndexInShell / numberOfPointsInShell

            point = ScanPoint(
                index=index,
                position_x_m=radiusInMeters * numpy.cos(thetaInRadians),
                position_y_m=radiusInMeters * numpy.sin(thetaInRadians),
            )
            pointList.append(point)

        return PositionSequence(pointList)
