from __future__ import annotations
from abc import ABC, abstractmethod


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

    def clearObservers(self) -> None:
        self._observerList.clear()

    def notifyObservers(self) -> None:
        for observer in self._observerList:
            observer.update(self)
