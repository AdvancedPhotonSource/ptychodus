from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Generic, Optional, Protocol, TypeVar
import logging

from ..api.observer import Observable, Observer

logger = logging.getLogger(__name__)


class RepositoryItem(Protocol):

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
        pass

    @property
    def canSelect(self) -> bool:
        '''indicates whether item can be selected'''
        pass

    def addObserver(self, observer: Observer) -> None:
        '''adds an observer'''
        pass

    def removeObserver(self, observer: Observer) -> None:
        '''removes an observer'''
        pass


T = TypeVar('T', bound=RepositoryItem)


class ItemRepository(Mapping[str, T], Observable):

    def __init__(self) -> None:
        super().__init__()
        self._itemDict: dict[str, T] = dict()
        self._nameList: list[str] = list()

    def __iter__(self) -> Iterator[str]:
        return iter(self._nameList)

    def __getitem__(self, name: str) -> T:
        return self._itemDict[name]

    def __len__(self) -> int:
        return len(self._nameList)

    def getNameItemTupleByIndex(self, index: int) -> tuple[str, T]:
        name = self._nameList[index]
        return name, self._itemDict[name]

    def insertItem(self, item: T) -> str:
        uniqueName = item.nameHint
        index = 0

        while uniqueName in self._itemDict:
            index += 1
            uniqueName = f'{item.nameHint}-{index}'

        self._itemDict[uniqueName] = item
        self._nameList.append(uniqueName)
        self.notifyObservers()
        return uniqueName

    def removeItem(self, name: str) -> None:
        try:
            item = self._itemDict.pop(name)
        except KeyError:
            pass

        try:
            self._nameList.remove(name)
        except ValueError:
            pass

        self.notifyObservers()


class RepositoryItemSettingsDelegate(Generic[T]):

    @abstractmethod
    def syncFromSettings(self) -> str:
        pass

    @abstractmethod
    def syncToSettings(self, item: T) -> None:
        pass


class SelectedRepositoryItem(Generic[T], Observable, Observer):

    def __init__(self, repository: ItemRepository[T],
                 settingsDelegate: RepositoryItemSettingsDelegate[T],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._repository = repository
        self._settingsDelegate = settingsDelegate
        self._reinitObservable = reinitObservable
        self._name = str()
        self._item: Optional[T] = None

    @classmethod
    def createInstance(cls, repository: ItemRepository[T],
                       settingsDelegate: RepositoryItemSettingsDelegate[T],
                       reinitObservable: Observable) -> SelectedRepositoryItem[T]:
        name = settingsDelegate.syncFromSettings()
        item = cls(repository, settingsDelegate, reinitObservable)
        item.selectItem(name)
        repository.addObserver(item)
        reinitObservable.addObserver(item)
        return item

    def getSelectableNames(self) -> Sequence[str]:
        return [name for name, item in self._repository.items() if item.canSelect]

    def getSelectedName(self) -> str:
        return self._name

    def getSelectedItem(self) -> Optional[T]:
        return self._item

    def canSelectItem(self, name: str) -> bool:
        try:
            item = self._repository[name]
        except KeyError:
            return False
        else:
            return item.canSelect

    def deselectItem(self) -> None:
        if self._item is not None:
            self._item.removeObserver(self)
            self._item = None

    def selectItem(self, name: str) -> None:
        if self._name == name:
            return

        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Failed to select \"{name}\"!')
            return

        if item.canSelect:
            if self._item is not None:
                self._item.removeObserver(self)

            self._item = item
            self._name = name
            self._item.addObserver(self)

            self._syncToSettings()
        else:
            logger.error(f'Failed to select \"{name}\"!')

        self.notifyObservers()

    def _recoverIfSelectedItemRemovedFromRepository(self) -> None:
        if self._name not in self._repository:
            try:
                name = next(iter(self._repository))
            except StopIteration:
                pass
            else:
                self.selectItem(name)

    def _syncFromSettings(self) -> None:
        name = self._settingsDelegate.syncFromSettings()
        self.selectItem(name)

    def _syncToSettings(self) -> None:
        if self._item is not None:
            self._settingsDelegate.syncToSettings(self._item)
        else:
            logger.error('Failed to sync null item to settings!')

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncToSettings()
        elif observable is self._repository:
            self._recoverIfSelectedItemRemovedFromRepository()
        elif observable is self._reinitObservable:
            self._syncFromSettings()
