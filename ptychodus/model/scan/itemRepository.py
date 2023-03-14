from __future__ import annotations
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass

from ...api.observer import Observable, Observer
from ...api.scan import Scan
from ..itemRepository import ItemRepository
from .settings import ScanSettings


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


@dataclass(frozen=True)
class ScanRepositoryItemPresenter:
    name: str
    initializer: str
    numberOfPoints: int


ScanRepository = ItemRepository[ScanRepositoryItem]


class ScanRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ScanRepository) -> None:
        super().__init__()
        self._repository = repository
        self._nameList: list[str] = list()

    @classmethod
    def createInstance(cls, repository: ScanRepository) -> ScanRepositoryPresenter:
        presenter = cls(repository)
        presenter._updateNameList()
        repository.addObserver(presenter)
        return presenter

    def __getitem__(self, index: int) -> ScanRepositoryItemPresenter:
        name = self._nameList[index]
        item = self._repository[name]
        return ScanRepositoryItemPresenter(
            name=name,
            initializer=item.initializer,
            numberOfPoints=len(item),
        )

    def __len__(self) -> int:
        return len(self._nameList)

    def canRemoveScan(self, name: str) -> bool:
        return self._repository.canRemoveItem(name)

    def removeScan(self, name: str) -> None:
        self._repository.removeItem(name)

    def _updateNameList(self) -> None:
        self._nameList = list(self._repository.keys())
        self._nameList.sort()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateNameList()
