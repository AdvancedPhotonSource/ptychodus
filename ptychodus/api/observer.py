from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic, TypeVar

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


class SequenceObserver(Generic[T], ABC):

    @abstractmethod
    def handleItemInserted(self, index: int, item: T) -> None:
        pass

    @abstractmethod
    def handleItemChanged(self, index: int, item: T) -> None:
        pass

    @abstractmethod
    def handleItemRemoved(self, index: int, item: T) -> None:
        pass


class ObservableSequence(Sequence[T]):

    def __init__(self) -> None:
        self._observerList: list[SequenceObserver[T]] = list()

    def addObserver(self, observer: SequenceObserver[T]) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: SequenceObserver[T]) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def notifyObserversItemInserted(self, index: int, item: T) -> None:
        for observer in self._observerList:
            observer.handleItemInserted(index, item)

    def notifyObserversItemChanged(self, index: int, item: T) -> None:
        for observer in self._observerList:
            observer.handleItemChanged(index, item)

    def notifyObserversItemRemoved(self, index: int, item: T) -> None:
        for observer in self._observerList:
            observer.handleItemRemoved(index, item)
