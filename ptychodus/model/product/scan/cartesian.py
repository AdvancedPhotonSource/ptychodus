from __future__ import annotations
from enum import IntEnum

import numpy

from ptychodus.api.scan import Scan, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


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
        return self.value & 1 != 0

    @property
    def isTriangular(self) -> bool:
        return self.value & 2 != 0

    @property
    def isEquilateral(self) -> bool:
        return self.value & 4 != 0


class CartesianScanBuilder(ScanBuilder):
    def __init__(self, variant: CartesianScanVariant, settings: ScanSettings) -> None:
        super().__init__(settings, variant.name.lower())
        self._variant = variant
        self._settings = settings

        self.numberOfPointsX = settings.numberOfPointsX.copy()
        self._addParameter('number_of_points_x', self.numberOfPointsX)

        self.numberOfPointsY = settings.numberOfPointsY.copy()
        self._addParameter('number_of_points_y', self.numberOfPointsY)

        self.stepSizeXInMeters = settings.stepSizeXInMeters.copy()
        self._addParameter('step_size_x_m', self.stepSizeXInMeters)

        self.stepSizeYInMeters = settings.stepSizeYInMeters.copy()
        self._addParameter('step_size_y_m', self.stepSizeYInMeters)

    def copy(self) -> CartesianScanBuilder:
        builder = CartesianScanBuilder(self._variant, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value.getValue())

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
