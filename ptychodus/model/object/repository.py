from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ...api.object import Object
from ...api.observer import Observable, Observer

logger = logging.getLogger(__name__)


class ObjectBuilder(ABC, Observable):

    @abstractmethod
    def build(self) -> Object:
        pass


class ObjectRepositoryItem(Observable, Observer):

    def __init__(self, object_: Object) -> None:
        super().__init__()
        self._object = object_  # FIXME handle pixel size = 0, center point?
        self._builder: ObjectBuilder | None = None

    def getObject(self) -> Object:
        return self._object

    def setObject(self, object_: Object) -> None:
        self._object = object_
        self._builder = None
        self.notifyObservers()

    def rebuild(self) -> None:
        if self._builder is None:
            logger.error('Missing object builder!')
            return

        try:
            object_ = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize object!')
        else:
            self._object = object_
            self.notifyObservers()

    def getBuilder(self) -> ObjectBuilder | None:
        return self._builder

    def setBuilder(self, builder: ObjectBuilder) -> None:
        if self._builder is not None:
            self._builder.removeObserver(self)

        self._builder = builder
        builder.addObserver(self)
        self.rebuild()

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self.rebuild()


# TODO class ObjectInitializer(ABC, Observable):
# TODO     '''ABC for plugins that can initialize objects'''
# TODO
# TODO     @property
# TODO     @abstractmethod
# TODO     def simpleName(self) -> str:
# TODO         '''returns a unique name that is appropriate for a settings file'''
# TODO         pass
# TODO
# TODO     @property
# TODO     @abstractmethod
# TODO     def displayName(self) -> str:
# TODO         '''returns a unique name that is prettified for visual display'''
# TODO         pass
# TODO
# TODO     @abstractmethod
# TODO     def syncFromSettings(self, settings: ObjectSettings) -> None:
# TODO         '''synchronizes initializer state from settings'''
# TODO         pass
# TODO
# TODO     @abstractmethod
# TODO     def syncToSettings(self, settings: ObjectSettings) -> None:
# TODO         '''synchronizes initializer state to settings'''
# TODO         pass
# TODO
# TODO     @abstractmethod
# TODO     def __call__(self) -> Object:
# TODO         '''produces an initial object guess'''
# TODO         pass
# TODO
# TODO
# TODO class ObjectRepositoryItem(Observable, Observer):
# TODO     '''container for items that can be stored in a object repository'''
# TODO     SIMPLE_NAME: Final[str] = 'FromMemory'
# TODO     DISPLAY_NAME: Final[str] = 'From Memory'
# TODO
# TODO     def __init__(self, nameHint: str) -> None:
# TODO         super().__init__()
# TODO         self._nameHint = nameHint
# TODO         self._object = Object()
# TODO         self._initializer: ObjectInitializer | None = None
# TODO
# TODO     @property
# TODO     def nameHint(self) -> str:
# TODO         '''returns a name hint that is appropriate for a settings file'''
# TODO         return self._nameHint
# TODO
# TODO     def getObject(self) -> Object:
# TODO         return self._object
# TODO
# TODO     def setObject(self, object_: Object) -> None:
# TODO         self._initializer = None
# TODO         self._object = object_
# TODO         self.notifyObservers()
# TODO
# TODO     def reinitialize(self) -> None:
# TODO         if self._initializer is None:
# TODO             logger.error('Missing object initializer!')
# TODO             return
# TODO
# TODO         try:
# TODO             object_ = self._initializer()
# TODO         except Exception:
# TODO             logger.exception('Failed to reinitialize object!')
# TODO             return
# TODO
# TODO         self._object = object_
# TODO         self.notifyObservers()
# TODO
# TODO     def getInitializerSimpleName(self) -> str:
# TODO         return self.SIMPLE_NAME if self._initializer is None else self._initializer.simpleName
# TODO
# TODO     def getInitializerDisplayName(self) -> str:
# TODO         return self.DISPLAY_NAME if self._initializer is None else self._initializer.displayName
# TODO
# TODO     def getInitializer(self) -> ObjectInitializer | None:
# TODO         return self._initializer
# TODO
# TODO     def setInitializer(self, initializer: ObjectInitializer) -> None:
# TODO         if self._initializer is not None:
# TODO             self._initializer.removeObserver(self)
# TODO
# TODO         self._initializer = initializer
# TODO         initializer.addObserver(self)
# TODO         self.reinitialize()
# TODO
# TODO     def update(self, observable: Observable) -> None:
# TODO         if observable is self._initializer:
# TODO             self.reinitialize()
# TODO
# TODO
# TODO ObjectRepository = ItemRepository[ObjectRepositoryItem]
