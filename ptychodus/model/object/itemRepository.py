from abc import abstractmethod, abstractproperty

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from ...api.observer import Observable
from ..itemRepository import ItemRepository
from .settings import ObjectSettings


class ObjectRepositoryItem(Observable):
    '''ABC for items that can be stored in a object repository'''

    @abstractproperty
    def name(self) -> str:
        '''returns a unique name'''
        pass

    @abstractproperty
    def initializer(self) -> str:
        '''returns a unique initializer name'''
        pass

    @abstractproperty
    def canActivate(self) -> bool:
        '''indicates whether item can be made active'''
        pass

    @abstractmethod
    def syncFromSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes item state from settings'''
        pass

    @abstractmethod
    def syncToSettings(self, settings: ObjectSettings) -> None:
        '''synchronizes item state to settings'''
        pass

    @abstractmethod
    def getDataType(self) -> str:
        '''returns the array data type'''
        pass

    @abstractmethod
    def getExtentInPixels(self) -> ImageExtent:
        '''returns the array width and height'''
        pass

    @abstractmethod
    def getSizeInBytes(self) -> int:
        '''returns the array size in bytes'''
        pass

    @abstractmethod
    def getArray(self) -> ObjectArrayType:
        '''returns the array data'''
        pass


ObjectRepository = ItemRepository[ObjectRepositoryItem]
