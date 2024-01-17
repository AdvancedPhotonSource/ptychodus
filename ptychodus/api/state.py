from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Generic, TypeVar
import logging

import numpy
import numpy.typing

from .patterns import DiffractionPatternArrayType, DiffractionPatternIndexes

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffractionPatternStateData:
    indexes: DiffractionPatternIndexes
    array: DiffractionPatternArrayType


class StatefulCore(Generic[T]):
    '''interface for module cores that have state data'''

    @abstractmethod
    def getStateData(self) -> T:
        pass

    @abstractmethod
    def setStateData(self, stateData: T, stateFilePath: Path) -> None:
        pass


class StateDataRegistry:
    FILE_FILTER: Final[str] = 'NumPy Zipped Archive (*.npz)'
    DATA_INDEXES: Final[str] = 'dataIndexes'
    DATA_ARRAY: Final[str] = 'data'

    def __init__(self, dataCore: StatefulCore[DiffractionPatternStateData]) -> None:
        self._dataCore = dataCore

    def openStateData(self, filePath: Path) -> None:
        '''reads a state data from file'''
        logger.debug(f'Reading state data from \"{filePath}\"')
        stateData = numpy.load(filePath)

        try:
            dataState = DiffractionPatternStateData(
                indexes=stateData[self.DATA_INDEXES],
                array=stateData[self.DATA_ARRAY],
            )
        except KeyError:
            logger.debug('Diffraction pattern state not found.')
        else:
            self._dataCore.setStateData(dataState, filePath)

    def saveStateData(self, filePath: Path, *, restartable: bool) -> None:
        '''save state data to file'''
        logger.debug(f'Writing state data to \"{filePath}\" [restartable={restartable}]')
        stateData: dict[str, numpy.typing.NDArray[Any]] = dict()

        if restartable:
            try:
                dataState = self._dataCore.getStateData()
            except ValueError:
                logger.warning('Failed to save data state.')
            else:
                stateData[self.DATA_INDEXES] = dataState.indexes
                stateData[self.DATA_ARRAY] = dataState.array

        numpy.savez(filePath, **stateData)
