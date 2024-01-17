from __future__ import annotations
from enum import IntEnum

import numpy

from ...api.parametric import ParametricBase
from ...api.scan import ScanPoint
from .metrics import ScanMetrics


class ScanPointTransform(IntEnum):
    '''transformations to negate or swap scan point coordinates'''
    PXPY = 0x0
    MXPY = 0x1
    PXMY = 0x2
    MXMY = 0x3
    PYPX = 0x4
    PYMX = 0x5
    MYPX = 0x6
    MYMX = 0x7

    @property
    def negateX(self) -> bool:
        '''indicates whether the x coordinate is negated'''
        return self.value & 1 != 0

    @property
    def negateY(self) -> bool:
        '''indicates whether the y coordinate is negated'''
        return self.value & 2 != 0

    @property
    def swapXY(self) -> bool:
        '''indicates whether the x and y coordinates are swapped'''
        return self.value & 4 != 0

    @property
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        xp = '-x' if self.negateX else '+x'
        yp = '-y' if self.negateY else '+y'
        return f'{yp}{xp}' if self.swapXY else f'{xp}{yp}'

    @property
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        xp = '\u2212x' if self.negateX else '\u002Bx'
        yp = '\u2212y' if self.negateY else '\u002By'
        return f'({yp}, {xp})' if self.swapXY else f'({xp}, {yp})'

    def __call__(self, point: ScanPoint) -> ScanPoint:
        '''transforms a scan point'''
        xp = -point.positionXInMeters if self.negateX else point.positionXInMeters
        yp = -point.positionYInMeters if self.negateY else point.positionYInMeters
        return ScanPoint(point.index, yp, xp) if self.swapXY else ScanPoint(point.index, xp, yp)


class ScanTransformer(ParametricBase):

    def __init__(self, rng: numpy.random.Generator, metrics: ScanMetrics) -> None:
        super().__init__('Transform')
        self._rng = rng
        self._metrics = metrics
        self.transform = ScanPointTransform.PXPY  # FIXME -> parameter
        self.isOverrideCentroidXEnabled = self._registerBooleanParameter(
            'IsOverrideCentroidXEnabled', False)
        self.overrideCentroidXInMeters = self._registerRealParameter(
            'OverrideCentroidXInMeters',
            0.,
        )
        self.isOverrideCentroidYEnabled = self._registerBooleanParameter(
            'IsOverrideCentroidYEnabled', False)
        self.overrideCentroidYInMeters = self._registerRealParameter(
            'OverrideCentroidYInMeters',
            0.,
        )
        self.jitterRadiusInMeters = self._registerRealParameter(
            'JitterRadiusInMeters',
            0.,
            minimum=0.,
        )

    def __call__(self, point: ScanPoint) -> ScanPoint:  # FIXME
        midpoint = self._metrics.getMidpointInMeters()
        transformedPoint = self.transform(point)
        posX = transformedPoint.positionXInMeters
        posY = transformedPoint.positionYInMeters
        jitterRadiusInMeters = self.jitterRadiusInMeters.getValue()

        if self.isOverrideCentroidXEnabled:
            posX += self.overrideCentroidXInMeters.getValue() - midpoint.x

        if self.isOverrideCentroidYEnabled:
            posY += self.overrideCentroidYInMeters.getValue() - midpoint.y

        if jitterRadiusInMeters > 0.:
            rad = self._rng.uniform()
            dirX = self._rng.normal()
            dirY = self._rng.normal()

            scalar = jitterRadiusInMeters * numpy.sqrt(rad / (dirX**2 + dirY**2))
            posX += scalar * dirX
            posY += scalar * dirY

        return ScanPoint(point.index, posX, posY)
