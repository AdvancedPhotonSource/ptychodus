from __future__ import annotations
from collections.abc import Iterator, Mapping
from typing import Generic, Protocol, TypeVar
import logging

from ..api.observer import Observable

logger = logging.getLogger(__name__)


class RepositoryItem(Protocol):

    @property
    def name(self) -> str:
        pass

    def clearObservers(self) -> None:
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
