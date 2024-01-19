from collections.abc import Callable, Iterator, Mapping, Sequence

import numpy

from ...api.parametric import ParameterRepository
from ...api.scan import ScanPoint
from .metrics import ScanMetrics


class ScanPointTransform(ParameterRepository):

    def __init__(self, affineAX: float, affineAY: float, affineAT: float, affineBX: float,
                 affineBY: float, affineBT: float, jitterRadiusInMeters: float,
                 rng: numpy.random.Generator) -> None:
        super().__init__('Transform')

        self.affineAX = self._registerRealParameter('AffineAX', affineAX)
        self.affineAY = self._registerRealParameter('AffineAY', affineAY)
        self.affineAT = self._registerRealParameter('AffineAT', affineAT)

        self.affineBX = self._registerRealParameter('AffineBX', affineBX)
        self.affineBY = self._registerRealParameter('AffineBY', affineBY)
        self.affineBT = self._registerRealParameter('AffineBT', affineBT)

        self.jitterRadiusInMeters = self._registerRealParameter(
            'JitterRadiusInMeters',
            jitterRadiusInMeters,
            minimum=0.,
        )
        self._rng = rng

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
            radsq = rad * rad

            while True:
                dX = self._rng.uniform()
                dY = self._rng.uniform()

                if dX * dX + dY * dY < radsq:
                    posX += dX
                    posY += dY
                    break

        return ScanPoint(point.index, posX, posY)


class ScanPointTransformFactory(Mapping[str, ScanPointTransform]):

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng
        self._transformations: Mapping[str, Callable[[], ScanPointTransform]] = {
            '\u002BX\u002BY': self._createPXPY,
            '\u2212X\u002BY': self._createMXPY,
            '\u002BX\u2212Y': self._createPXMY,
            '\u2212X\u2212Y': self._createMXMY,
            '\u002BY\u002BX': self._createPYPX,
            '\u002BY\u2212X': self._createPYMX,
            '\u2212Y\u002BX': self._createMYPX,
            '\u2212Y\u2212X': self._createMYMX,
        }

    def _createPXPY(self) -> ScanPointTransform:
        return ScanPointTransform(+1., 0., 0., 0., +1., 0., 0., self._rng)

    def _createMXPY(self) -> ScanPointTransform:
        return ScanPointTransform(-1., 0., 0., 0., +1., 0., 0., self._rng)

    def _createPXMY(self) -> ScanPointTransform:
        return ScanPointTransform(+1., 0., 0., 0., -1., 0., 0., self._rng)

    def _createMXMY(self) -> ScanPointTransform:
        return ScanPointTransform(-1., 0., 0., 0., -1., 0., 0., self._rng)

    def _createPYPX(self) -> ScanPointTransform:
        return ScanPointTransform(0., +1., 0., +1., 0., 0., 0., self._rng)

    def _createPYMX(self) -> ScanPointTransform:
        return ScanPointTransform(0., +1., 0., -1., 0., 0., 0., self._rng)

    def _createMYPX(self) -> ScanPointTransform:
        return ScanPointTransform(0., -1., 0., +1., 0., 0., 0., self._rng)

    def _createMYMX(self) -> ScanPointTransform:
        return ScanPointTransform(0., -1., 0., -1., 0., 0., 0., self._rng)

    def createDefaultTransform(self) -> ScanPointTransform:
        return self._createPXPY()

    def __iter__(self) -> Iterator[str]:
        return iter(self._transformations)

    def __getitem__(self, name: str) -> ScanPointTransform:
        try:
            factory = self._transformations[name]
        except KeyError as exc:
            raise KeyError(f'Unknown scan point transform \"{name}\"!') from exc

        return factory()

    def __len__(self) -> int:
        return len(self._transformations)
