from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy.typing

StateDataKeyType = str
StateDataValueType = numpy.typing.NDArray[Any]
StateDataType = Mapping[StateDataKeyType, StateDataValueType]


class StatefulCore(ABC):
    '''interface for module cores that have state data'''

    @abstractmethod
    def getStateData(self, *, restartable: bool) -> StateDataType:
        pass

    @abstractmethod
    def setStateData(self, state: StateDataType) -> None:
        pass
