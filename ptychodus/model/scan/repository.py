from __future__ import annotations
from abc import abstractmethod, abstractproperty
from collections.abc import Iterator, Mapping
import logging

from ...api.observer import Observable
from ...api.scan import Scan
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ContiguousScanIterator(Iterator[int]):

    def __init__(self, scan: Scan) -> None:
        self._scan = scan
        self._index = 0

    def __iter__(self) -> ContiguousScanIterator:
        return self

    def __next__(self) -> int:
        if self._index < len(self._scan):
            index = self._index
            self._index += 1
            return index

        raise StopIteration


class ScanRepositoryItem(Scan):
    '''ABC for items that can be stored in a scan repository'''

    @abstractproperty
    def category(self) -> str:
        '''returns a unique category for organizing scan positions'''
        pass

    @abstractproperty
    def variant(self) -> str:
        '''returns a unique variant name'''
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


class ScanRepository(Mapping[str, ScanRepositoryItem], Observable):

    def __init__(self) -> None:
        super().__init__()
        self._itemDict: dict[str, ScanRepositoryItem] = dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self._itemDict)

    def __getitem__(self, name: str) -> ScanRepositoryItem:
        return self._itemDict[name]

    def __len__(self) -> int:
        return len(self._itemDict)

    def insertItem(self, item: ScanRepositoryItem) -> str:
        uniqueName = item.name
        index = 0

        while uniqueName in self._itemDict:
            index += 1
            uniqueName = f'{item.name}-{index}'

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
                item.clearObservers()
        else:
            logger.debug(f'Cannot remove item \"{name}\"')

        self.notifyObservers()
