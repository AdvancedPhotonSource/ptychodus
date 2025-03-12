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
        self._add_parameter('affine_ax', self.affineAX)

        self.affineAY = settings.affineTransformAY.copy()
        self._add_parameter('affine_ay', self.affineAY)

        self.affineATInMeters = settings.affineTransformATInMeters.copy()
        self._add_parameter('affine_at_m', self.affineATInMeters)

        self.affineBX = settings.affineTransformBX.copy()
        self._add_parameter('affine_bx', self.affineBX)

        self.affineBY = settings.affineTransformBY.copy()
        self._add_parameter('affine_by', self.affineBY)

        self.affineBTInMeters = settings.affineTransformBTInMeters.copy()
        self._add_parameter('affine_bt_m', self.affineBTInMeters)

        self.jitterRadiusInMeters = settings.jitterRadiusInMeters.copy()
        self._add_parameter('jitter_radius_m', self.jitterRadiusInMeters)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def copy(self) -> ScanPointTransform:
        transform = ScanPointTransform(self._rng, self._settings)

        for key, value in self.parameters().items():
            transform.parameters()[key].set_value(value.get_value())

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
            self.affineAY.set_value(-1 if self.negateY(index) else +1)
            self.affineBX.set_value(-1 if self.negateX(index) else +1)
            self.affineAX.set_value(0)
            self.affineBY.set_value(0)
        else:
            self.affineAX.set_value(-1 if self.negateX(index) else +1)
            self.affineBY.set_value(-1 if self.negateY(index) else +1)
            self.affineAY.set_value(0)
            self.affineBX.set_value(0)

    def __call__(self, point: ScanPoint) -> ScanPoint:
        ax = self.affineAX.get_value()
        ay = self.affineAY.get_value()
        at_m = self.affineATInMeters.get_value()

        bx = self.affineBX.get_value()
        by = self.affineBY.get_value()
        bt_m = self.affineBTInMeters.get_value()

        posX = ax * point.position_x_m + ay * point.position_y_m + at_m
        posY = bx * point.position_x_m + by * point.position_y_m + bt_m

        rad = self.jitterRadiusInMeters.get_value()

        if rad > 0.0:
            while True:
                dX = self._rng.uniform()
                dY = self._rng.uniform()

                if dX * dX + dY * dY < 1.0:
                    posX += dX * rad
                    posY += dY * rad
                    break

        return ScanPoint(point.index, posX, posY)
