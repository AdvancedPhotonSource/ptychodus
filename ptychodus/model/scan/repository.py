from __future__ import annotations
from collections.abc import Iterator, Mapping
import logging

from ...api.observer import Observable
from .repositoryItem import ScanRepositoryItem

logger = logging.getLogger(__name__)


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

    def insertItem(self, item: ScanRepositoryItem) -> None:
        uniqueName = item.name
        index = 0

        while uniqueName in self._itemDict:
            index += 1
            uniqueName = f'{item.name}-{index}'

        self._itemDict[uniqueName] = item
        self.notifyObservers()

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
