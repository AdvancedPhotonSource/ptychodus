from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import ScanPoint

from .settings import ScanSettings


class ScanPointTransform(ParameterGroup):
    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings

        self.affineAX = settings.affineTransformAX.copy()
        self._addParameter('affine_ax', self.affineAX)

        self.affineAY = settings.affineTransformAY.copy()
        self._addParameter('affine_ay', self.affineAY)

        self.affineATInMeters = settings.affineTransformATInMeters.copy()
        self._addParameter('affine_at_m', self.affineATInMeters)

        self.affineBX = settings.affineTransformBX.copy()
        self._addParameter('affine_bx', self.affineBX)

        self.affineBY = settings.affineTransformBY.copy()
        self._addParameter('affine_by', self.affineBY)

        self.affineBTInMeters = settings.affineTransformBTInMeters.copy()
        self._addParameter('affine_bt_m', self.affineBTInMeters)

        self.jitterRadiusInMeters = settings.jitterRadiusInMeters.copy()
        self._addParameter('jitter_radius_m', self.jitterRadiusInMeters)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

    def copy(self) -> ScanPointTransform:
        transform = ScanPointTransform(self._rng, self._settings)

        for key, value in self.parameters().items():
            transform.parameters()[key].setValue(value)

        return transform

    @staticmethod
    def negateX(preset: int) -> bool:
        return preset & 0x1 != 0x0

    @staticmethod
    def negateY(preset: int) -> bool:
        return preset & 0x2 != 0x0

    @staticmethod
    def swapXY(preset: int) -> bool:
        return preset & 0x4 != 0x0

    def labelsForPresets(self) -> Iterator[str]:
        for index in range(8):
            xp = '\u2212x' if self.negateX(index) else '\u002bx'
            yp = '\u2212y' if self.negateY(index) else '\u002by'
            fxy = f'{yp}, {xp}' if self.swapXY(index) else f'{xp}, {yp}'
            yield f'(x, y) \u2192 ({fxy})'

    def applyPresets(self, index: int) -> None:
        if self.swapXY(index):
            self.affineAY.setValue(-1 if self.negateY(index) else +1)
            self.affineBX.setValue(-1 if self.negateX(index) else +1)
            self.affineAX.setValue(0)
            self.affineBY.setValue(0)
        else:
            self.affineAX.setValue(-1 if self.negateX(index) else +1)
            self.affineBY.setValue(-1 if self.negateY(index) else +1)
            self.affineAY.setValue(0)
            self.affineBX.setValue(0)

    def __call__(self, point: ScanPoint) -> ScanPoint:
        ax = self.affineAX.getValue()
        ay = self.affineAY.getValue()
        at_m = self.affineATInMeters.getValue()

        bx = self.affineBX.getValue()
        by = self.affineBY.getValue()
        bt_m = self.affineBTInMeters.getValue()

        posX = ax * point.positionXInMeters + ay * point.positionYInMeters + at_m
        posY = bx * point.positionXInMeters + by * point.positionYInMeters + bt_m

        rad = self.jitterRadiusInMeters.getValue()

        if rad > 0.0:
            while True:
                dX = self._rng.uniform()
                dY = self._rng.uniform()

                if dX * dX + dY * dY < 1.0:
                    posX += dX * rad
                    posY += dY * rad
                    break

        return ScanPoint(point.index, posX, posY)
