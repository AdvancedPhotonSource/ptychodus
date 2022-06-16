from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import typing


class IPCMessage(ABC):

    @classmethod
    @abstractproperty
    def messageType(self) -> int:
        pass

    @classmethod
    @abstractmethod
    def fromDict(self, values: dict[str, typing.Any]) -> IPCMessage:
        pass

    def toDict(self) -> dict[str, typing.Any]:
        return {'messageType': self.messageType}
