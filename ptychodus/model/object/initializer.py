from abc import ABC, abstractmethod, abstractproperty

from ...api.object import ObjectArrayType
from ...api.observer import Observable
from .settings import ObjectSettings


class ObjectInitializer(Observable, ABC):
    '''ABC for plugins that can initialize objects'''

    @abstractmethod
    def syncFromSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes initializer state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes initializer state to settings'''
        settings.initializer.value = self.simpleName

    @abstractproperty
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        return ''.join(self.displayName.split())

    @abstractmethod
    def __call__(self) -> ObjectArrayType:
        '''produces an initial object guess'''
        pass
