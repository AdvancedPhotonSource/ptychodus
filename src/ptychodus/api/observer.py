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
    def _update(self, observable: Observable) -> None:
        pass


class Observable:
    def __init__(self) -> None:
        self._observer_list: list[Observer] = list()
        self._block_notifications = False
        self._pending_notify = False

    def add_observer(self, observer: Observer) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: Observer) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def block_notifications(self, block: bool) -> None:
        self._block_notifications = block

        if self._pending_notify:
            self.notify_observers()
            self._pending_notify = False

    def notify_observers(self) -> None:
        if self._block_notifications:
            self._pending_notify = True
            return

        for observer in self._observer_list:
            observer._update(self)


class SequenceObserver(Generic[T], ABC):
    @abstractmethod
    def handle_item_inserted(self, index: int, item: T) -> None:
        pass

    @abstractmethod
    def handle_item_changed(self, index: int, item: T) -> None:
        pass

    @abstractmethod
    def handle_item_removed(self, index: int, item: T) -> None:
        pass


class ObservableSequence(Sequence[T]):
    def __init__(self) -> None:
        self._observer_list: list[SequenceObserver[T]] = list()

    def add_observer(self, observer: SequenceObserver[T]) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: SequenceObserver[T]) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def notify_observers_item_inserted(self, index: int, item: T) -> None:
        for observer in self._observer_list:
            observer.handle_item_inserted(index, item)

    def notify_observers_item_changed(self, index: int, item: T) -> None:
        for observer in self._observer_list:
            observer.handle_item_changed(index, item)

    def notify_observers_item_removed(self, index: int, item: T) -> None:
        for observer in self._observer_list:
            observer.handle_item_removed(index, item)
