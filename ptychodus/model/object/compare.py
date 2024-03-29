from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Final
import logging

from ...api.object import Object
from ...api.observer import Observable, Observer
from ...api.plot import FourierRingCorrelation
from .repository import ObjectInitializer, ObjectRepositoryItem
from .settings import ObjectSettings
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class CompareObjectInitializer(ObjectInitializer, Observer):
    SIMPLE_NAME: Final[str] = 'Compare'
    DISPLAY_NAME: Final[str] = 'Compare'

    def __init__(self, sizer: ObjectSizer, repository: Mapping[str, ObjectRepositoryItem]) -> None:
        super().__init__()
        self._sizer = sizer
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

    def __call__(self) -> Object:
        object1 = self._item1.getObject()
        object2 = self._item2.getObject()

        if not object1.hasSameShape(object2):
            logger.warning('Shape mismatch!')

        difference = object2.copy()
        difference.setArray(object1.getArray() - object2.getArray())
        return difference

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

    def getFourierRingCorrelation(self) -> FourierRingCorrelation:
        # TODO support multiple layers
        return FourierRingCorrelation.calculate(
            self._item1.getObject().getLayer(0),
            self._item2.getObject().getLayer(0),
            self._sizer.getPixelGeometry(),
        )

    def update(self, observable: Observable) -> None:
        if observable is self._item1:
            self.notifyObservers()
        elif observable is self._item2:
            self.notifyObservers()
