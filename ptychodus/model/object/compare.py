from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Final
import logging

import numpy

from ...api.object import ObjectArrayType
from ...api.observer import Observable, Observer
from .repository import ObjectInitializer, ObjectRepositoryItem
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class CompareObjectInitializer(ObjectInitializer, Observer):
    SIMPLE_NAME: Final[str] = 'Compare'
    DISPLAY_NAME: Final[str] = 'Compare'

    def __init__(self, repository: Mapping[str, ObjectRepositoryItem]) -> None:
        super().__init__()
        self._repository = repository
        self._name1 = str()
        self._item1 = ObjectRepositoryItem('')
        self._name2 = str()
        self._item2 = ObjectRepositoryItem('')

        if repository:
            name = self._getNameFallback()
            self.setName1(name)
            self.setName2(name)

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        pass

    def syncToSettings(self, settings: ObjectSettings) -> None:
        pass

    def __call__(self) -> ObjectArrayType:
        array1 = self._item1.getArray()
        array2 = self._item2.getArray()

        if array1.shape != array2.shape:
            logger.warning(f'Shape mismatch: {array1.shape} vs {array2.shape}!')
            return array1

        return array1 - array2

    def getComparableNames(self) -> Sequence[str]:
        return [name for name, item in self._repository.items() \
                if not isinstance(item.getInitializer(), CompareObjectInitializer)]

    def _getNameFallback(self) -> str:
        try:
            name = next(iter(self._repository))
        except StopIteration:
            name = str()

        return name

    def getName1(self) -> str:
        return self._name1 if self._name1 in self._repository else self._getNameFallback()

    def setName1(self, name: str) -> None:
        try:
            item = self._repository[name]
        except KeyError:
            logger.warning(f'Failed to get \"{name}\" from repository!')
            return

        if self._name1 != name:
            self._name1 = name
            self._item1.removeObserver(self)
            self._item1 = item
            self._item1.addObserver(self)
            self.notifyObservers()

    def getName2(self) -> str:
        return self._name2 if self._name2 in self._repository else self._getNameFallback()

    def setName2(self, name: str) -> None:
        try:
            item = self._repository[name]
        except KeyError:
            logger.warning(f'Failed to get \"{name}\" from repository!')
            return

        if self._name2 != name:
            self._name2 = name
            self._item2.removeObserver(self)
            self._item2 = item
            self._item2.addObserver(self)
            self.notifyObservers()

    def getSpatialFrequency(self) -> Sequence[float]:
        n = 100
        return [i / (n - 1) for i in range(n)]

    def getFourierRingCorrelation(self) -> Sequence[float]:
        x = self.getSpatialFrequency()
        y = numpy.sin(2 * numpy.pi * numpy.array(x))  # FIXME
        return list(y)

    def update(self, observable: Observable) -> None:
        if observable is self._item1:
            self.notifyObservers()
        elif observable is self._item2:
            self.notifyObservers()
