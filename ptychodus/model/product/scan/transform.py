from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.scan import ScanPoint

from .settings import ScanSettings


class ScanPointTransform(ParameterRepository):

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings) -> None:
        super().__init__('transform')
        self._rng = rng
        self._settings = settings

        self.affineAX = self._registerRealParameter('affine_ax',
                                                    float(settings.affineTransformAX.value))
        self.affineAY = self._registerRealParameter('affine_ay',
                                                    float(settings.affineTransformAY.value))
        self.affineATInMeters = self._registerRealParameter(
            'affine_at_m', float(settings.affineTransformATInMeters.value))

        self.affineBX = self._registerRealParameter('affine_bx',
                                                    float(settings.affineTransformBX.value))
        self.affineBY = self._registerRealParameter('affine_by',
                                                    float(settings.affineTransformBY.value))
        self.affineBTInMeters = self._registerRealParameter(
            'affine_bt_m', float(settings.affineTransformBTInMeters.value))

        self.jitterRadiusInMeters = self._registerRealParameter(
            'jitter_radius_m',
            float(settings.jitterRadiusInMeters.value),
            minimum=0.,
        )

    def copy(self) -> ScanPointTransform:
        transform = ScanPointTransform(self._rng, self._settings)
        transform.affineAX.setValue(self.affineAX.getValue())
        transform.affineAY.setValue(self.affineAY.getValue())
        transform.affineATInMeters.setValue(self.affineATInMeters.getValue())
        transform.affineBX.setValue(self.affineBX.getValue())
        transform.affineBY.setValue(self.affineBY.getValue())
        transform.affineBTInMeters.setValue(self.affineBTInMeters.getValue())
        transform.jitterRadiusInMeters.setValue(self.jitterRadiusInMeters.getValue())
        return transform

    @staticmethod
    def negateX(preset: int) -> bool:
        return (preset & 0x1 != 0x0)

    @staticmethod
    def negateY(preset: int) -> bool:
        return (preset & 0x2 != 0x0)

    @staticmethod
    def swapXY(preset: int) -> bool:
        return (preset & 0x4 != 0x0)

    def labelsForPresets(self) -> Iterator[str]:
        for index in range(8):
            xp = '\u2212x' if self.negateX(index) else '\u002Bx'
            yp = '\u2212y' if self.negateY(index) else '\u002By'
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

        if rad > 0.:
            while True:
                dX = self._rng.uniform()
                dY = self._rng.uniform()

                if dX * dX + dY * dY < 1.:
                    posX += dX * rad
                    posY += dY * rad
                    break

        return ScanPoint(point.index, posX, posY)
