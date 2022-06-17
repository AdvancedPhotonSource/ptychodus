from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import Any


class RPCMessage(ABC):

    @classmethod
    @abstractproperty
    def messageType(cls) -> int:
        '''returns a unique integer'''
        pass

    @classmethod
    @abstractmethod
    def fromDict(cls, values: dict[str, Any]) -> RPCMessage:
        '''creates and populates a message class from a dictionary'''
        pass

    def toDict(self) -> dict[str, Any]:
        '''creates and populates a dictionary from a message class'''
        return {'messageType': self.messageType}


class RPCMessageHandler(ABC):

    @abstractmethod
    def handleMessage(self, message: RPCMessage) -> None:
        '''performs action using information in message'''
        pass
