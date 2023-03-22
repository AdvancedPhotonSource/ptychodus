from __future__ import annotations
from abc import abstractmethod, abstractproperty
from collections.abc import Iterator
from decimal import Decimal
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.scan import Scan, ScanIndexFilter, ScanPoint, ScanPointTransform
from ..itemRepository import ItemRepository
from .indexFilters import ScanIndexFilterFactory
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ScanRepositoryItem(Scan):
    '''ABC for items that can be stored in a scan repository'''

    @abstractproperty
    def initializer(self) -> str:
        '''returns a unique initializer name'''
        pass

    @abstractproperty
    def canActivate(self) -> bool:
        '''indicates whether item can be made active'''
        pass

    @abstractmethod
    def syncFromSettings(self, settings: ScanSettings) -> None:
        '''synchronizes item state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ScanSettings) -> None:
        '''synchronizes item state to settings'''
        pass


class TransformedScanRepositoryItem(ScanRepositoryItem, Observer):

    def __init__(self, rng: numpy.random.Generator, item: ScanRepositoryItem,
                 indexFilterFactory: ScanIndexFilterFactory) -> None:
        super().__init__()
        self._rng = rng
        self._item = item
        self._indexFilterFactory = indexFilterFactory
        self._indexFilter = indexFilterFactory.create('All')
        self._transform = ScanPointTransform.PXPY
        self._jitterRadiusInMeters = Decimal()
        self._centroid = ScanPoint(Decimal(), Decimal())
        self._name: Optional[str] = None

    @classmethod
    def createInstance(
            cls, rng: numpy.random.Generator, item: ScanRepositoryItem,
            indexFilterFactory: ScanIndexFilterFactory) -> TransformedScanRepositoryItem:
        transformedItem = cls(rng, item, indexFilterFactory)
        item.addObserver(transformedItem)
        return transformedItem

    @property
    def name(self) -> str:
        return self._item.name if self._name is None else self._name

    @name.setter
    def name(self, name: str) -> None:
        if self._name != name:
            self._name = name
            self.notifyObservers()

    @property
    def initializer(self) -> str:
        return self._item.initializer

    @property
    def canActivate(self) -> bool:
        return self._item.canActivate

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._indexFilter = self._indexFilterFactory.create(settings.indexFilter.value)
        self._transform = ScanPointTransform.fromSimpleName(settings.transform.value)
        self_jitterRadiusInMeters = settings.jitterRadiusInMeters.value
        self._centroid = ScanPoint(settings.centroidXInMeters.value,
                                   settings.centroidYInMeters.value)
        self._item.syncFromSettings(settings)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.indexFilter.value = self._indexFilter.name
        settings.transform.value = self._transform.simpleName
        settings.jitterRadiusInMeters.value = self._jitterRadiusInMeters
        settings.centroidXInMeters.value = self._centroid.x
        settings.centroidYInMeters.value = self._centroid.y
        self._item.syncToSettings(settings)

    def __iter__(self) -> Iterator[int]:
        it = iter(self._item)

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

        point = self._item[index]

        if self._jitterRadiusInMeters > Decimal():
            rad = Decimal(repr(self._rng.uniform()))
            dirX = Decimal(repr(self._rng.normal()))
            dirY = Decimal(repr(self._rng.normal()))

            scalar = self._jitterRadiusInMeters * (rad / (dirX**2 + dirY**2)).sqrt()
            point = ScanPoint(point.x + scalar * dirX, point.y + scalar * dirY)

        point = self._transform(point)

        return ScanPoint(
            self._centroid.x + point.x,
            self._centroid.y + point.y,
        )

    def __len__(self) -> int:
        return sum(1 for index in iter(self))

    @property
    def untransformed(self) -> ScanRepositoryItem:
        return self._item

    def getIndexFilterNameList(self) -> list[str]:
        return self._indexFilterFactory.getIndexFilterNameList()

    def getIndexFilterName(self) -> str:
        return self._indexFilter.name

    def setIndexFilterByName(self, name: str) -> None:
        indexFilter = self._indexFilterFactory.create(name)

        if self._indexFilter != indexFilter:
            self._indexFilter = indexFilter
            self.notifyObservers()

    def getTransformNameList(self) -> list[str]:
        return [transform.displayName for transform in ScanPointTransform]

    def getTransformName(self) -> str:
        return self._transform.displayName

    def setTransformByName(self, name: str) -> None:
        nameLower = name.casefold()

        for transform in ScanPointTransform:
            if nameLower == transform.displayName.casefold():
                self.setTransform(transform)
                return

        logger.error(f'Unknown scan point transform \"{name}\"!')

    def getTransform(self) -> ScanPointTransform:
        '''gets the scan point transform'''
        return self._transform

    def setTransform(self, transform: ScanPointTransform) -> None:
        '''sets the scan point transform'''
        if self._transform != transform:
            self._transform = transform
            self.notifyObservers()

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
        if observable is self._item:
            self.notifyObservers()


ScanRepository = ItemRepository[TransformedScanRepositoryItem]
