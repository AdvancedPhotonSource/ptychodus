from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from typing import Generic, Protocol, TypeVar
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

    def __iter__(self) -> Iterator[str]:
        return iter(self._itemDict)

    def __getitem__(self, name: str) -> T:
        return self._itemDict[name]

    def __len__(self) -> int:
        return len(self._itemDict)

    def insertItem(self, item: T) -> str:
        uniqueName = item.nameHint
        index = 0

        while uniqueName in self._itemDict:
            index += 1
            uniqueName = f'{item.nameHint}-{index}'

        self._itemDict[uniqueName] = item
        self.notifyObservers()
        return uniqueName

    def canRemoveItem(self, name: str) -> bool:
        return len(self._itemDict) > 1

    def removeItem(self, name: str) -> None:
        if self.canRemoveItem(name):
            try:
                item = self._itemDict.pop(name)
            except KeyError:
                pass
        else:
            logger.debug(f'Cannot remove item \"{name}\"')

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
        self._name = next(iter(self._repository.keys()))
        self._item = next(iter(self._repository.values()))

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

    def getSelectedName(self) -> str:
        return self._name

    def getSelectedItem(self) -> T:
        return self._item

    def canSelectItem(self, name: str) -> bool:
        try:
            item = self._repository[name]
        except KeyError:
            return False
        else:
            return item.canSelect

    def selectItem(self, name: str) -> None:
        if self._name == name:
            return

        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Failed to select \"{name}\"!')
            return

        if not item.canSelect:
            logger.error(f'Failed to select \"{name}\"!')
            return

        self._item.removeObserver(self)
        self._item = item
        self._name = name
        self._item.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    def _recoverIfSelectedItemRemovedFromRepository(self) -> None:
        if self._name not in self._repository:
            self.selectItem(next(iter(self._repository)))

    def _syncFromSettings(self) -> None:
        name = self._settingsDelegate.syncFromSettings()
        self.selectItem(name)

    def _syncToSettings(self) -> None:
        self._settingsDelegate.syncToSettings(self._item)

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncToSettings()
        elif observable is self._repository:
            self._recoverIfSelectedItemRemovedFromRepository()
        elif observable is self._reinitObservable:
            self._syncFromSettings()
