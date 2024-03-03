from enum import IntEnum

import numpy

from ...api.scan import Scan, ScanPoint
from .builder import ScanBuilder


class CartesianScanVariant(IntEnum):
    RASTER = 0x0
    SNAKE = 0x1
    CENTERED_RASTER = 0x2
    CENTERED_SNAKE = 0x3
    HEXAGONAL_RASTER = 0x6
    HEXAGONAL_SNAKE = 0x7

    @property
    def isSnaked(self) -> bool:
        return (self.value & 1 != 0)

    @property
    def isCentered(self) -> bool:
        return (self.value & 2 != 0)

    @property
    def isHexagonal(self) -> bool:
        return (self.value & 4 != 0)

    def getName(self) -> str:
        nameList: list[str] = list()

        if self.isHexagonal:
            # FIXME verify hexagonal
            nameList.append('Hexagonal')
        elif self.isCentered:
            nameList.append('Centered')

        if self.isSnaked:
            nameList.append('Snake')
        else:
            nameList.append('Raster')

        return ' '.join(nameList)


class CartesianScanBuilder(ScanBuilder):

    def __init__(self, *, variant: CartesianScanVariant) -> None:
        super().__init__(variant.getName())
        self._variant = variant
        self.stepSizeXInMeters = self._registerRealParameter('StepSizeXInMeters', 1e-6, minimum=0.)
        self.stepSizeYInMeters = self._registerRealParameter('StepSizeYInMeters', 1e-6, minimum=0.)
        self.numberOfPointsX = self._registerIntegerParameter('NumberOfPointsX', 10, minimum=0)
        self.numberOfPointsY = self._registerIntegerParameter('NumberOfPointsY', 10, minimum=0)

    def build(self) -> Scan:
        nx = self.numberOfPointsX.getValue()
        ny = self.numberOfPointsY.getValue()
        dx = self.stepSizeXInMeters.getValue()
        dy = dx * numpy.sqrt(0.75) if self._variant.isHexagonal \
                else self.stepSizeYInMeters.getValue()

        pointList: list[ScanPoint] = list()

        for index in range(nx * ny):
            y, x = divmod(index, nx)

            if self._variant.isSnaked:
                if y & 1:
                    x = nx - 1 - x

            cx = (nx - 1) / 2
            cy = (ny - 1) / 2

            xf = (x - cx) * dx
            yf = (y - cy) * dy

            if self._variant.isCentered:
                if y & 1:
                    xf += dx / 4
                else:
                    xf -= dx / 4

            point = ScanPoint(
                index=index,
                positionXInMeters=xf,
                positionYInMeters=yf,
            )
            pointList.append(point)

        return Scan(pointList)
