from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from typing import Generic, Protocol, TypeVar
import logging

from ..api.observer import Observable, Observer

logger = logging.getLogger(__name__)


class RepositoryItem(Protocol):

    @property
    def nameHint(self) -> str:
        '''returns a name hint that is appropriate for a settings file'''
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

    def insertItem(self, item: T | None) -> str | None:
        if item is None:
            logger.error('Refusing to add null item to repository!')
            return None

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
            self._itemDict.pop(name)
        except KeyError:
            pass

        try:
            self._nameList.remove(name)
        except ValueError:
            pass

        self.notifyObservers()


class RepositoryItemSettingsDelegate(ABC, Generic[T]):

    @abstractmethod
    def syncFromSettings(self) -> str | None:
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
        self._item: T | None = None

    @classmethod
    def createInstance(cls, repository: ItemRepository[T],
                       settingsDelegate: RepositoryItemSettingsDelegate[T],
                       reinitObservable: Observable) -> SelectedRepositoryItem[T]:
        item = cls(repository, settingsDelegate, reinitObservable)
        repository.addObserver(item)
        reinitObservable.addObserver(item)
        return item

    def getSelectableNames(self) -> Sequence[str]:
        return list(self._repository.keys())

    def getSelectedName(self) -> str:
        return self._name

    def getSelectedItem(self) -> T | None:
        return self._item

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

        if self._item is not None:
            self._item.removeObserver(self)

        self._item = item
        self._name = name
        self._item.addObserver(self)

        self._syncToSettings()
        self.notifyObservers()

    def _recoverIfSelectedItemRemovedFromRepository(self) -> None:
        if self._name not in self._repository:
            try:
                name = next(iter(self._repository))
            except StopIteration:
                self.deselectItem()
            else:
                self.selectItem(name)

    def _syncFromSettings(self) -> None:
        name = self._settingsDelegate.syncFromSettings()

        if name is not None:
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
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self._syncFromSettings()
