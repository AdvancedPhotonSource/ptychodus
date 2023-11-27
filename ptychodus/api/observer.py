from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

__all__ = [
    'Observer',
    'Observable',
    'SequenceObserver',
    'ObservableSequence',
]

T = TypeVar('T')


class Observer(ABC):

    @abstractmethod
    def update(self, observable: Observable) -> None:
        pass


class Observable:

    def __init__(self) -> None:
        self._observerList: list[Observer] = list()

    def addObserver(self, observer: Observer) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: Observer) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def notifyObservers(self) -> None:
        for observer in self._observerList:
            observer.update(self)


class SequenceObserver(ABC):

    @abstractmethod
    def handleItemInserted(self, index: int) -> None:
        pass

    @abstractmethod
    def handleItemChanged(self, index: int) -> None:
        pass

    @abstractmethod
    def handleItemRemoved(self, index: int) -> None:
        pass


class ObservableSequence(Sequence[T]):

    def __init__(self) -> None:
        self._observerList: list[SequenceObserver] = list()

    def addObserver(self, observer: SequenceObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: SequenceObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def notifyObserversItemInserted(self, index: int) -> None:
        for observer in self._observerList:
            observer.handleItemInserted(index)

    def notifyObserversItemChanged(self, index: int) -> None:
        for observer in self._observerList:
            observer.handleItemChanged(index)

    def notifyObserversItemRemoved(self, index: int) -> None:
        for observer in self._observerList:
            observer.handleItemRemoved(index)
