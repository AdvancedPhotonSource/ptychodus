from __future__ import annotations
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from ...api.observer import Observable, Observer
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
    def getExtent(self) -> ImageExtent:
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


@dataclass(frozen=True)
class ObjectRepositoryItemPresenter:
    name: str
    initializer: str
    dataType: str
    extentInPixels: ImageExtent
    sizeInBytes: int


ObjectRepository = ItemRepository[ObjectRepositoryItem]


class ObjectRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ObjectRepository) -> None:
        super().__init__()
        self._repository = repository
        self._nameList: list[str] = list()

    @classmethod
    def createInstance(cls, repository: ObjectRepository) -> ObjectRepositoryPresenter:
        presenter = cls(repository)
        presenter._updateNameList()
        repository.addObserver(presenter)
        return presenter

    def __getitem__(self, index: int) -> ObjectRepositoryItemPresenter:
        name = self._nameList[index]
        item = self._repository[name]
        return ObjectRepositoryItemPresenter(
            name=name,
            initializer=item.initializer,
            dataType=item.getDataType(),
            extentInPixels=item.getExtent(),
            sizeInBytes=item.getSizeInBytes(),
        )

    def __len__(self) -> int:
        return len(self._nameList)

    def canRemoveObject(self, name: str) -> bool:
        return self._repository.canRemoveItem(name)

    def removeObject(self, name: str) -> None:
        self._repository.removeItem(name)

    def _updateNameList(self) -> None:
        self._nameList = list(self._repository.keys())
        self._nameList.sort()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateNameList()
