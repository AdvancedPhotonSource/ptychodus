from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
import logging

import numpy
import numpy.typing

logger = logging.getLogger(__name__)

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


class StateDataRegistry:

    def __init__(self, statefulCores: Iterable[StatefulCore]) -> None:
        self._statefulCores = statefulCores

    def saveStateData(self, filePath: Path, *, restartable: bool) -> None:
        # TODO document file format
        # TODO include cost function values
        logger.debug(f'Writing state data to \"{filePath}\" [restartable={restartable}]')
        data: dict[StateDataKeyType, StateDataValueType] = dict()

        for core in self._statefulCores:
            data.update(core.getStateData(restartable=restartable))

        numpy.savez(filePath, **data)

    def openStateData(self, filePath: Path) -> None:
        logger.debug(f'Reading state data from \"{filePath}\"')
        data = numpy.load(filePath)
        data['restartFilePath'] = numpy.array(list(str(filePath.resolve())))

        for core in self._statefulCores:
            core.setStateData(data)
