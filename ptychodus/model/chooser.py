from __future__ import annotations
from dataclasses import dataclass
from typing import Generic, TypeVar
import logging

from .observer import Observable

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyEntry(Generic[T]):
    simpleName: str
    displayName: str
    strategy: T


class StrategyChooser(Generic[T], Observable):
    def __init__(self, defaultEntry: StrategyEntry[T]) -> None:
        super().__init__()
        self._entryList: list[StrategyEntry[T]] = [defaultEntry]
        self._entry = defaultEntry

    @classmethod
    def createFromList(cls, entryList: list[StrategyEntry[T]]) -> StrategyChooser:
        chooser = cls(entryList[0])
        chooser._entryList = entryList.copy()
        return chooser

    def addStrategy(self, entry: StrategyEntry[T]) -> None:
        self._entryList.insert(0, entry)

    def setToDefault(self) -> None:
        self._setEntry(self._entryList[-1])

    def getSimpleNameList(self) -> list[str]:
        return [entry.simpleName for entry in self._entryList]

    def setFromSimpleName(self, name: str) -> None:
        try:
            entry = next(entry for entry in self._entryList
                         if name.casefold() == entry.simpleName.casefold())
        except StopIteration:
            logger.debug(f'Invalid strategy simple name \"{name}\"')
            return

        self._setEntry(entry)

    def getDisplayNameList(self) -> list[str]:
        return [entry.displayName for entry in self._entryList]

    def setFromDisplayName(self, name: str) -> None:
        try:
            entry = next(entry for entry in self._entryList if name == entry.displayName)
        except StopIteration:
            logger.debug(f'Invalid strategy display name \"{name}\"')
            return

        self._setEntry(entry)

    def getCurrentSimpleName(self) -> str:
        return self._entry.simpleName

    def getCurrentDisplayName(self) -> str:
        return self._entry.displayName

    def getCurrentStrategy(self) -> T:
        return self._entry.strategy

    def _setEntry(self, entry: StrategyEntry[T]) -> None:
        if self._entry is not entry:
            self._entry = entry
            self.notifyObservers()
