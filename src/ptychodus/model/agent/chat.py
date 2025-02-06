from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import overload


class ChatRole(Enum):
    HUMAN = auto()
    AI = auto()


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


class ChatObserver(ABC):
    @abstractmethod
    def handle_new_message(self, message: ChatMessage, index: int) -> None:
        pass

    @abstractmethod
    def handle_chat_cleared(self) -> None:
        pass


class ChatHistory(Sequence[ChatMessage]):
    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []
        self._observers: list[ChatObserver] = []

    @overload
    def __getitem__(self, index: int) -> ChatMessage: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ChatMessage]: ...

    def __getitem__(self, index: int | slice) -> ChatMessage | Sequence[ChatMessage]:
        return self._messages[index]

    def __len__(self) -> int:
        return len(self._messages)

    def add_observer(self, observer: ChatObserver) -> None:
        if observer not in self._observers:
            self._observers.append(observer)

    def add_message(self, message: ChatMessage) -> None:
        index = len(self._messages)
        self._messages.append(message)

        for observer in self._observers:
            observer.handle_new_message(message, index)

    def clear(self) -> None:
        self._messages.clear()

        for observer in self._observers:
            observer.handle_chat_cleared()
