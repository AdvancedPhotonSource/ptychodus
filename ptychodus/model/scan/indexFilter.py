from abc import ABC, abstractmethod
from collections.abc import Sequence
import logging

from ...api.observer import Observable

__all__ = [
    'SelectableScanIndexFilter',
]

logger = logging.getLogger(__name__)


class ScanIndexFilter(ABC):
    '''filters scan points by index'''

    @property
    @abstractmethod
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @property
    @abstractmethod
    def displayName(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def __call__(self, index: int) -> bool:
        '''include scan point if true, remove otherwise'''
        pass


class AllScanIndexesFilter(ScanIndexFilter):

    @property
    def displayName(self) -> str:
        return self.simpleName

    @property
    def simpleName(self) -> str:
        return 'All'

    def __call__(self, index: int) -> bool:
        return True


class OddScanIndexesFilter(ScanIndexFilter):

    @property
    def displayName(self) -> str:
        return self.simpleName

    @property
    def simpleName(self) -> str:
        return 'Odd'

    def __call__(self, index: int) -> bool:
        return (index % 2 != 0)


class EvenScanIndexesFilter(ScanIndexFilter):

    @property
    def displayName(self) -> str:
        return self.simpleName

    @property
    def simpleName(self) -> str:
        return 'Even'

    def __call__(self, index: int) -> bool:
        return (index % 2 == 0)


class SelectableScanIndexFilter(ScanIndexFilter, Observable):

    def __init__(self) -> None:
        super().__init__()
        self._availableFilters = [
            AllScanIndexesFilter(),
            OddScanIndexesFilter(),
            EvenScanIndexesFilter(),
        ]
        self._indexFilter: ScanIndexFilter = self._availableFilters[0]

    def getSelectableFilters(self) -> Sequence[str]:
        return [indexFilter.displayName for indexFilter in self._availableFilters]

    def selectFilterFromSimpleName(self, name: str) -> None:
        nameFold = name.casefold()

        for indexFilter in self._availableFilters:
            if nameFold == indexFilter.simpleName.casefold():
                self._indexFilter = indexFilter
                self.notifyObservers()
                return

        logger.error(f'Unknown scan index filter \"{name}\"!')

    def selectFilterFromDisplayName(self, name: str) -> None:
        nameFold = name.casefold()

        for indexFilter in self._availableFilters:
            if nameFold == indexFilter.displayName.casefold():
                self._indexFilter = indexFilter
                self.notifyObservers()
                return

        logger.error(f'Unknown scan index filter \"{name}\"!')

    @property
    def simpleName(self) -> str:
        return self._indexFilter.simpleName

    @property
    def displayName(self) -> str:
        return self._indexFilter.displayName

    def __call__(self, index: int) -> bool:
        return self._indexFilter(index)
