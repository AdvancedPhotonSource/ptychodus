from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import typing


class RPCMessage(ABC):

    @classmethod
    @abstractproperty
    def messageType(self) -> int:
        '''returns a unique integer'''
        pass

    @classmethod
    @abstractmethod
    def fromDict(self, values: dict[str, typing.Any]) -> RPCMessage:
        '''creates and populates a message class from a dictionary'''
        pass

    def toDict(self) -> dict[str, typing.Any]:
        '''creates and populates a dictionary from a message class'''
        return {'messageType': self.messageType}
