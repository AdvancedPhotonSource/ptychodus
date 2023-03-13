from __future__ import annotations
from abc import abstractmethod, abstractproperty

from ...api.scan import Scan
from ..itemRepository import ItemRepository
from .settings import ScanSettings


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


ScanRepository = ItemRepository[ScanRepositoryItem]
