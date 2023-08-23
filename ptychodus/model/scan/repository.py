from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Optional
import logging
import sys

import numpy

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.scan import Scan, ScanPoint, TabularScan
from ..itemRepository import ItemRepository
from .indexFilter import SelectableScanIndexFilter
from .settings import ScanSettings
from .transform import SelectableScanPointTransform

logger = logging.getLogger(__name__)


class ScanInitializer(ABC, Observable):

    @property
    @abstractmethod
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @property
    @abstractmethod
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def syncFromSettings(self, settings: ScanSettings) -> None:
        '''synchronizes item state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ScanSettings) -> None:
        '''synchronizes item state to settings'''
        pass

    @abstractmethod
    def __call__(self) -> Scan:
        pass


class ScanRepositoryItem(Scan, Observable, Observer):

    def __init__(self,
                 rng: numpy.random.Generator,
                 nameHint: str,
                 scan: Optional[Scan] = None) -> None:
        super().__init__()
        self._rng = rng
        self._nameHint = nameHint
        self._scan: Scan = TabularScan({}) if scan is None else scan
        self._initializer: Optional[ScanInitializer] = None
        self._transform = SelectableScanPointTransform()
        self._transform.addObserver(self)
        self._indexFilter = SelectableScanIndexFilter()
        self._indexFilter.addObserver(self)
        self._overrideCentroidXEnabled = False
        self._overrideCentroidXInMeters = Decimal()
        self._overrideCentroidYEnabled = False
        self._overrideCentroidYInMeters = Decimal()
        self._jitterRadiusInMeters = Decimal()

        self._cachedNumberOfPoints = 0
        self._cachedLengthInMeters = 0.
        self._cachedCentroidXInMeters = 0.
        self._cachedCentroidYInMeters = 0.
        self._cachedSizeInBytes = 0

        self._updateCacheAndNotifyObservers()

    def _updateCacheAndNotifyObservers(self) -> None:
        pointList = [point for point in self.values()]
        lengthInMeters = 0.

        try:
            point = pointList[0]
        except IndexError:
            self._cachedCentroidXInMeters = 0.
            self._cachedCentroidYInMeters = 0.
            logger.debug('Scan is empty!')
        else:
            rangeX = Interval[float](point.x, point.x)
            rangeY = Interval[float](point.y, point.y)

            for pointA, pointB in zip(pointList[:-1], pointList[1:]):
                rangeX = rangeX.hull(pointB.x)
                rangeY = rangeY.hull(pointB.y)

                dx = pointB.x - pointA.x
                dy = pointB.y - pointA.y
                lengthInMeters += numpy.hypot(dx, dy)

            self._cachedCentroidXInMeters = rangeX.midrange
            self._cachedCentroidYInMeters = rangeY.midrange

        self._cachedNumberOfPoints = len(pointList)
        self._cachedLengthInMeters = lengthInMeters
        self._cachedSizeInBytes = self._getSizeInBytes()
        self.notifyObservers()

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    def reinitialize(self) -> None:
        '''reinitializes the scan point sequence'''
        if self._initializer is None:
            logger.error('Missing scan initializer!')
            return

        try:
            self._scan = self._initializer()
        except Exception:
            logger.exception('Failed to reinitialize scan!')
        else:
            self._updateCacheAndNotifyObservers()

    def getInitializerSimpleName(self) -> str:
        return 'FromMemory' if self._initializer is None else self._initializer.simpleName

    def getInitializer(self) -> Optional[ScanInitializer]:
        return self._initializer

    def setInitializer(self, initializer: ScanInitializer) -> None:
        if self._initializer is not None:
            self._initializer.removeObserver(self)

        self._initializer = initializer
        initializer.addObserver(self)
        self.reinitialize()

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._indexFilter.selectFilterByName(settings.indexFilter.value)
        self._transform.selectTransformByName(settings.transform.value)
        self._jitterRadiusInMeters = settings.jitterRadiusInMeters.value
        self._overrideCentroidXEnabled = settings.overrideCentroidXEnabled.value
        self._overrideCentroidXInMeters = settings.overrideCentroidXInMeters.value
        self._overrideCentroidYEnabled = settings.overrideCentroidYEnabled.value
        self._overrideCentroidYInMeters = settings.overrideCentroidYInMeters.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.indexFilter.value = self._indexFilter.simpleName
        settings.transform.value = self._transform.simpleName
        settings.jitterRadiusInMeters.value = self._jitterRadiusInMeters
        settings.overrideCentroidXEnabled.value = self._overrideCentroidXEnabled
        settings.overrideCentroidXInMeters.value = self._overrideCentroidXInMeters
        settings.overrideCentroidYEnabled.value = self._overrideCentroidYEnabled
        settings.overrideCentroidYInMeters.value = self._overrideCentroidYInMeters

    @property
    def untransformed(self) -> Scan:
        return self._scan

    def __iter__(self) -> Iterator[int]:
        it = iter(self._scan)

        while True:
            try:
                index = next(it)
            except StopIteration:
                break

            if self._indexFilter(index):
                yield index

    def __getitem__(self, index: int) -> ScanPoint:
        '''returns the jittered and transformed scan point'''
        if not self._indexFilter(index):
            raise KeyError

        point = self._transform(self._scan[index])
        posX = point.x
        posY = point.y

        if self.isOverrideCentroidXEnabled:
            posX += float(self._overrideCentroidXInMeters) - self._cachedCentroidXInMeters

        if self.isOverrideCentroidYEnabled:
            posY += float(self._overrideCentroidYInMeters) - self._cachedCentroidYInMeters

        if self._jitterRadiusInMeters > Decimal():
            rad = self._rng.uniform()
            dirX = self._rng.normal()
            dirY = self._rng.normal()

            scalar = float(self._jitterRadiusInMeters) * numpy.sqrt(rad / (dirX**2 + dirY**2))
            posX += scalar * dirX
            posY += scalar * dirY

        return ScanPoint(posX, posY)

    def __len__(self) -> int:
        return self._cachedNumberOfPoints

    def getLengthInMeters(self) -> float:
        return self._cachedLengthInMeters

    def _getSizeInBytes(self) -> int:
        sizeInBytes = sys.getsizeof(self._scan)

        for index, point in self._scan.items():
            sizeInBytes += sys.getsizeof(index)
            sizeInBytes += sys.getsizeof(point)

        return sizeInBytes

    def getSizeInBytes(self) -> int:
        return self._cachedSizeInBytes

    def getIndexFilterNameList(self) -> Sequence[str]:
        return self._indexFilter.getSelectableFilters()

    def getIndexFilterName(self) -> str:
        return self._indexFilter.displayName

    def setIndexFilterByName(self, name: str) -> None:
        self._indexFilter.selectFilterByName(name)

    def getTransformNameList(self) -> Sequence[str]:
        return self._transform.getSelectableTransforms()

    def getTransformName(self) -> str:
        return self._transform.displayName

    def setTransformByName(self, name: str) -> None:
        self._transform.selectTransformByName(name)

    def getJitterRadiusInMeters(self) -> Decimal:
        return self._jitterRadiusInMeters

    def setJitterRadiusInMeters(self, jitterRadiusInMeters: Decimal) -> None:
        if self._jitterRadiusInMeters != jitterRadiusInMeters:
            self._jitterRadiusInMeters = jitterRadiusInMeters
            self.notifyObservers()

    @property
    def isOverrideCentroidXEnabled(self) -> bool:
        return self._overrideCentroidXEnabled

    def setOverrideCentroidXEnabled(self, enabled: bool) -> None:
        if self._overrideCentroidXEnabled != enabled:
            self._overrideCentroidXEnabled = enabled
            self.notifyObservers()

    def getCentroidXInMeters(self) -> Decimal:
        return self._overrideCentroidXInMeters if self._overrideCentroidXEnabled \
                else Decimal(repr(self._cachedCentroidXInMeters))

    def setCentroidXInMeters(self, value: Decimal) -> None:
        if self._overrideCentroidXInMeters != value:
            self._overrideCentroidXInMeters = value
            self.notifyObservers()

    @property
    def isOverrideCentroidYEnabled(self) -> bool:
        return self._overrideCentroidYEnabled

    def setOverrideCentroidYEnabled(self, enabled: bool) -> None:
        if self._overrideCentroidYEnabled != enabled:
            self._overrideCentroidYEnabled = enabled
            self.notifyObservers()

    def getCentroidYInMeters(self) -> Decimal:
        return self._overrideCentroidYInMeters if self._overrideCentroidYEnabled \
                else Decimal(repr(self._cachedCentroidYInMeters))

    def setCentroidYInMeters(self, value: Decimal) -> None:
        if self._overrideCentroidYInMeters != value:
            self._overrideCentroidYInMeters = value
            self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable in (self._initializer, self._transform, self._indexFilter):
            self.reinitialize()


ScanRepository = ItemRepository[ScanRepositoryItem]
