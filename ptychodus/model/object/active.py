from __future__ import annotations
import logging

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from ...api.observer import Observable, Observer
from .itemFactory import ObjectRepositoryItemFactory
from .itemRepository import ObjectRepository, ObjectRepositoryItem
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ActiveObject(Observable, Observer):

    def __init__(self, settings: ObjectSettings, factory: ObjectRepositoryItemFactory,
                 repository: ObjectRepository, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._factory = factory
        self._repository = repository
        self._reinitObservable = reinitObservable
        self._item: ObjectRepositoryItem = factory.createRandomItem()
        self._name = str()

    @classmethod
    def createInstance(cls, settings: ObjectSettings, factory: ObjectRepositoryItemFactory,
                       repository: ObjectRepository, reinitObservable: Observable) -> ActiveObject:
        object_ = cls(settings, factory, repository, reinitObservable)
        object_._syncFromSettings()
        repository.addObserver(object_)
        reinitObservable.addObserver(object_)
        return object_

    @property
    def name(self) -> str:
        return self._name

    def canActivateObject(self, name: str) -> bool:
        item = self._repository.get(name)

        if item is not None:
            return item.canActivate

        return False

    def setActiveObject(self, name: str) -> None:
        if self._name == name:
            return

        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        if not item.canActivate:
            logger.error(f'Failed to activate \"{name}\"!')
            return

        self._item.removeObserver(self)
        self._item = item
        self._name = name
        self._item.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    def getExtent(self) -> ImageExtent:
        return self._item.getExtent()

    def getArray(self) -> ObjectArrayType:
        return self._item.getArray()

    def _syncFromSettings(self) -> None:
        initializerName = self._settings.initializer.value
        item = self._factory.createItem(initializerName)

        if item is None:
            logger.error(f'Unknown object initializer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(item)
            self.setActiveObject(itemName)

    def _syncToSettings(self) -> None:
        self._settings.initializer.value = self._item.initializer
        self._item.syncToSettings(self._settings)
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncToSettings()
        elif observable is self._repository:
            pass  # FIXME do the right thing if the active object is removed
        elif observable is self._reinitObservable:
            self._syncFromSettings()
