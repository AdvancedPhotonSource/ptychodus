from __future__ import annotations
from collections.abc import Sequence
from enum import Enum
import logging

from ...api.observer import Observable
from ...api.scan import ScanPoint

__all__ = [
    'SelectableScanPointTransform',
]

logger = logging.getLogger(__name__)


class ScanPointTransform(Enum):
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
        xp = -point.x if self.negateX else point.x
        yp = -point.y if self.negateY else point.y
        return ScanPoint(yp, xp) if self.swapXY else ScanPoint(xp, yp)


class SelectableScanPointTransform(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._transform = ScanPointTransform.PXPY

    def getSelectableTransforms(self) -> Sequence[str]:
        return [transform.displayName for transform in ScanPointTransform]

    def selectTransformByName(self, name: str) -> None:
        namecf = name.casefold()

        for transform in ScanPointTransform:
            simpleMatch = (namecf == transform.simpleName.casefold())
            displayMatch = (namecf == transform.displayName.casefold())

            if simpleMatch or displayMatch:
                self._transform = transform
                self.notifyObservers()
                return

        logger.error(f'Unknown scan point transform \"{name}\"!')

    @property
    def simpleName(self) -> str:
        return self._transform.simpleName

    @property
    def displayName(self) -> str:
        return self._transform.displayName

    def __call__(self, point: ScanPoint) -> ScanPoint:
        return self._transform(point)
