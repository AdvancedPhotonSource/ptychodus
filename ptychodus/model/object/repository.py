from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Final
import logging

from ...api.object import Object
from ...api.observer import Observable, Observer
from ..itemRepository import ItemRepository
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectInitializer(ABC, Observable):
    '''ABC for plugins that can initialize objects'''

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
    def syncFromSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes initializer state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes initializer state to settings'''
        pass

    @abstractmethod
    def __call__(self) -> Object:
        '''produces an initial object guess'''
        pass


class ObjectRepositoryItem(Observable, Observer):
    '''container for items that can be stored in a object repository'''
    SIMPLE_NAME: Final[str] = 'FromMemory'
    DISPLAY_NAME: Final[str] = 'From Memory'

    def __init__(self, nameHint: str) -> None:
        super().__init__()
        self._nameHint = nameHint
        self._object = Object()
        self._initializer: ObjectInitializer | None = None

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        return self._nameHint

    def getObject(self) -> Object:
        return self._object

    def setObject(self, object_: Object) -> None:
        self._initializer = None
        self._object = object_
        self.notifyObservers()

    def reinitialize(self) -> None:
        if self._initializer is None:
            logger.error('Missing object initializer!')
            return

        try:
            object_ = self._initializer()
        except Exception:
            logger.exception('Failed to reinitialize object!')
            return

        self._object = object_
        self.notifyObservers()

    def getInitializerSimpleName(self) -> str:
        return self.SIMPLE_NAME if self._initializer is None else self._initializer.simpleName

    def getInitializerDisplayName(self) -> str:
        return self.DISPLAY_NAME if self._initializer is None else self._initializer.displayName

    def getInitializer(self) -> ObjectInitializer | None:
        return self._initializer

    def setInitializer(self, initializer: ObjectInitializer) -> None:
        if self._initializer is not None:
            self._initializer.removeObserver(self)

        self._initializer = initializer
        initializer.addObserver(self)
        self.reinitialize()

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self.reinitialize()


ObjectRepository = ItemRepository[ObjectRepositoryItem]
