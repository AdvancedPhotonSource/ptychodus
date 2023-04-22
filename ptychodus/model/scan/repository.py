from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator, Sequence
from decimal import Decimal
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.scan import Scan, ScanPoint, TabularScan
from ..itemRepository import ItemRepository
from .indexFilter import SelectableScanIndexFilter
from .settings import ScanSettings
from .transform import SelectableScanPointTransform

logger = logging.getLogger(__name__)


class ScanInitializer(ABC, Observable):

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
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
    '''container for items that can be stored in a scan repository'''

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
        self._jitterRadiusInMeters = Decimal()
        self._centroid = ScanPoint(Decimal(), Decimal())
        # FIXME scan distance and size in bytes

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    @property
    def canSelect(self) -> bool:
        '''indicates whether item can be selected'''
        return (self._initializer is not None)

    def reinitialize(self) -> None:
        '''reinitializes the scan point sequence'''
        if self._initializer is None:
            logger.error('Missing scan initializer!')
            return

        try:
            self._scan = self._initializer()
        except:
            logger.exception('Failed to reinitialize scan!')
        else:
            self.notifyObservers()

    def getInitializerSimpleName(self) -> str:
        return 'FromMemory' if self._initializer is None else self._initializer.simpleName

    def getInitializer(self) -> Optional[ScanInitializer]:
        '''returns the initializer'''
        return self._initializer

    def setInitializer(self, initializer: ScanInitializer) -> None:
        '''sets the initializer'''
        if self._initializer is not None:
            self._initializer.removeObserver(self)

        self._initializer = initializer
        initializer.addObserver(self)
        self.reinitialize()

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._indexFilter.selectFilterFromSimpleName(settings.indexFilter.value)
        self._transform.selectTransformFromSimpleName(settings.transform.value)
        self_jitterRadiusInMeters = settings.jitterRadiusInMeters.value
        self._centroid = ScanPoint(settings.centroidXInMeters.value,
                                   settings.centroidYInMeters.value)
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.indexFilter.value = self._indexFilter.simpleName
        settings.transform.value = self._transform.simpleName
        settings.jitterRadiusInMeters.value = self._jitterRadiusInMeters
        settings.centroidXInMeters.value = self._centroid.x
        settings.centroidYInMeters.value = self._centroid.y

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

        if self._jitterRadiusInMeters > Decimal():
            rad = Decimal(repr(self._rng.uniform()))
            dirX = Decimal(repr(self._rng.normal()))
            dirY = Decimal(repr(self._rng.normal()))

            scalar = self._jitterRadiusInMeters * (rad / (dirX**2 + dirY**2)).sqrt()
            point = ScanPoint(point.x + scalar * dirX, point.y + scalar * dirY)

        return ScanPoint(
            self._centroid.x + point.x,
            self._centroid.y + point.y,
        )

    def __len__(self) -> int:
        return sum(1 for index in iter(self))

    @property
    def untransformed(self) -> Scan:
        return self._scan

    def getIndexFilterNameList(self) -> Sequence[str]:
        return self._indexFilter.getSelectableFilters()

    def getIndexFilterName(self) -> str:
        return self._indexFilter.displayName

    def setIndexFilterByName(self, name: str) -> None:
        self._indexFilter.selectFilterFromDisplayName(name)

    def getTransformNameList(self) -> Sequence[str]:
        return self._transform.getSelectableTransforms()

    def getTransformName(self) -> str:
        return self._transform.displayName

    def setTransformByName(self, name: str) -> None:
        self._transform.selectTransformFromDisplayName(name)

    def getJitterRadiusInMeters(self) -> Decimal:
        '''gets the jitter radius'''
        return self._jitterRadiusInMeters

    def setJitterRadiusInMeters(self, jitterRadiusInMeters: Decimal) -> None:
        '''sets the jitter radius'''
        if self._jitterRadiusInMeters != jitterRadiusInMeters:
            self._jitterRadiusInMeters = jitterRadiusInMeters
            self.notifyObservers()

    def getCentroidXInMeters(self) -> Decimal:
        '''gets the x centroid'''
        return self._centroid.x

    def setCentroidXInMeters(self, value: Decimal) -> None:
        '''sets the x centroid'''
        if self._centroid.x != value:
            self._centroid = ScanPoint(value, self._centroid.y)
            self.notifyObservers()

    def getCentroidYInMeters(self) -> Decimal:
        '''gets the y centroid'''
        return self._centroid.y

    def setCentroidYInMeters(self, value: Decimal) -> None:
        '''sets the y centroid'''
        if self._centroid.y != value:
            self._centroid = ScanPoint(self._centroid.x, value)
            self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable in (self._initializer, self._transform, self._indexFilter):
            self.reinitialize()


ScanRepository = ItemRepository[ScanRepositoryItem]
