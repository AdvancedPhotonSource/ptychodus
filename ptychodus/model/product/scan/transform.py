from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.scan import ScanPoint


class ScanPointTransform(ParameterRepository):

    def __init__(self, rng: numpy.random.Generator) -> None:
        super().__init__('transform')
        self._rng = rng

        self.affineAX = self._registerRealParameter('affine_ax', 1.)
        self.affineAY = self._registerRealParameter('affine_ay', 0.)
        self.affineAT = self._registerRealParameter('affine_at', 0.)

        self.affineBX = self._registerRealParameter('affine_bx', 0.)
        self.affineBY = self._registerRealParameter('affine_by', 1.)
        self.affineBT = self._registerRealParameter('affine_bt', 0.)

        self.jitterRadiusInMeters = self._registerRealParameter(
            'jitter_radius_m',
            0.,
            minimum=0.,
        )

    def copy(self) -> ScanPointTransform:
        transform = ScanPointTransform(self._rng)
        transform.affineAX.setValue(self.affineAX.getValue())
        transform.affineAY.setValue(self.affineAY.getValue())
        transform.affineAT.setValue(self.affineAT.getValue())
        transform.affineBX.setValue(self.affineBX.getValue())
        transform.affineBY.setValue(self.affineBY.getValue())
        transform.affineBT.setValue(self.affineBT.getValue())
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
        at = self.affineAT.getValue()

        bx = self.affineBX.getValue()
        by = self.affineBY.getValue()
        bt = self.affineBT.getValue()

        posX = ax * point.positionXInMeters + ay * point.positionYInMeters + at
        posY = bx * point.positionXInMeters + by * point.positionYInMeters + bt

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
