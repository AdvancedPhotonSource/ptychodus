from __future__ import annotations
from enum import IntEnum

import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder


class CartesianScanVariant(IntEnum):
    RECTANGULAR_RASTER = 0x0
    RECTANGULAR_SNAKE = 0x1
    TRIANGULAR_RASTER = 0x2
    TRIANGULAR_SNAKE = 0x3
    SQUARE_RASTER = 0x4
    SQUARE_SNAKE = 0x5
    HEXAGONAL_RASTER = 0x6
    HEXAGONAL_SNAKE = 0x7

    @property
    def isSnaked(self) -> bool:
        return (self.value & 1 != 0)

    @property
    def isTriangular(self) -> bool:
        return (self.value & 2 != 0)

    @property
    def isEquilateral(self) -> bool:
        return (self.value & 4 != 0)

    def getDisplayName(self) -> str:
        if self is CartesianScanVariant.RECTANGULAR_RASTER:
            return 'Rectangular Raster'
        if self is CartesianScanVariant.RECTANGULAR_SNAKE:
            return 'Rectangular Snake'
        if self is CartesianScanVariant.TRIANGULAR_RASTER:
            return 'Triangular Raster'
        if self is CartesianScanVariant.TRIANGULAR_SNAKE:
            return 'Triangular Snake'
        if self is CartesianScanVariant.SQUARE_RASTER:
            return 'Square Raster'
        if self is CartesianScanVariant.SQUARE_SNAKE:
            return 'Square Snake'
        if self is CartesianScanVariant.HEXAGONAL_RASTER:
            return 'Hexagonal Raster'
        if self is CartesianScanVariant.HEXAGONAL_SNAKE:
            return 'Hexagonal Snake'

        return 'UNKNOWN'


class CartesianScanBuilder(ScanBuilder):

    def __init__(self, variant: CartesianScanVariant) -> None:
        super().__init__(variant.getDisplayName())
        self._variant = variant
        self.numberOfPointsX = self._registerIntegerParameter('NumberOfPointsX', 10, minimum=0)
        self.numberOfPointsY = self._registerIntegerParameter('NumberOfPointsY', 10, minimum=0)
        self.stepSizeXInMeters = self._registerRealParameter('StepSizeXInMeters', 1e-6, minimum=0.)
        self.stepSizeYInMeters = self._registerRealParameter('StepSizeYInMeters', 1e-6, minimum=0.)

    def copy(self) -> CartesianScanBuilder:
        builder = CartesianScanBuilder(self._variant)
        builder.numberOfPointsX.setValue(self.numberOfPointsX.getValue())
        builder.numberOfPointsY.setValue(self.numberOfPointsY.getValue())
        builder.stepSizeXInMeters.setValue(self.stepSizeXInMeters.getValue())
        builder.stepSizeYInMeters.setValue(self.stepSizeYInMeters.getValue())
        return builder

    @property
    def isEquilateral(self) -> bool:
        return self._variant.isEquilateral

    def build(self) -> Scan:
        nx = self.numberOfPointsX.getValue()
        ny = self.numberOfPointsY.getValue()
        dx = self.stepSizeXInMeters.getValue()

        if self._variant.isEquilateral:
            dy = dx

            if self._variant.isTriangular:
                dy *= numpy.sqrt(0.75)
        else:
            dy = self.stepSizeYInMeters.getValue()

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

            if self._variant.isTriangular:
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
