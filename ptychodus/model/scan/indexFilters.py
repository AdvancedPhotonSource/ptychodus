from collections.abc import Callable, Mapping
from typing import Final
import logging

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import ScanIndexFilter

__all__ = [
    'ScanIndexFilterFactory',
]

logger = logging.getLogger(__name__)


class AllScanIndexesFilter(ScanIndexFilter):
    NAME: Final[str] = 'All'

    @property
    def name(self) -> str:
        return self.NAME

    def __call__(self, index: int) -> bool:
        return True


class OddScanIndexesFilter(ScanIndexFilter):
    NAME: Final[str] = 'Odd'

    @property
    def name(self) -> str:
        return self.NAME

    def __call__(self, index: int) -> bool:
        return (index % 2 != 0)


class EvenScanIndexesFilter(ScanIndexFilter):
    NAME: Final[str] = 'Even'

    @property
    def name(self) -> str:
        return self.NAME

    def __call__(self, index: int) -> bool:
        return (index % 2 == 0)


class ScanIndexFilterFactory:

    def __init__(self) -> None:
        self._variants: Mapping[str, Callable[[], ScanIndexFilter]] = {
            AllScanIndexesFilter.NAME.casefold(): AllScanIndexesFilter,
            OddScanIndexesFilter.NAME.casefold(): OddScanIndexesFilter,
            EvenScanIndexesFilter.NAME.casefold(): EvenScanIndexesFilter,
        }

    def getIndexFilterNameList(self) -> list[str]:
        return [name.title() for name in self._variants]

    def create(self, name: str) -> ScanIndexFilter:
        try:
            indexFilter = self._variants[name.casefold()]
        except KeyError:
            logger.error(f'Unknown scan index filter \"{name}\"!')
            indexFilter = AllScanIndexesFilter

        return indexFilter()
