from __future__ import annotations
from abc import abstractmethod, abstractproperty
from collections.abc import Iterator

from ...api.scan import Scan
from .settings import ScanSettings


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
