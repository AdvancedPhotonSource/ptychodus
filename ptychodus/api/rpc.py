from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from typing import Any


class RPCMessage(ABC):
    '''interface for remote procedure call messages'''

    @classmethod
    @abstractproperty
    def procedure(cls) -> str:
        '''returns a unique name for the procedure'''
        pass

    @classmethod
    @abstractmethod
    def fromDict(cls, values: dict[str, Any]) -> RPCMessage:
        '''creates and populates a message class from a dictionary'''
        pass

    def toDict(self) -> dict[str, Any]:
        '''creates and populates a dictionary from a message class'''
        return {'procedure': self.procedure}


class RPCExecutor(ABC):
    '''interface for remote procedure call executors'''

    @abstractmethod
    def submit(self, message: RPCMessage) -> None:
        '''performs action using information in message'''
        pass
