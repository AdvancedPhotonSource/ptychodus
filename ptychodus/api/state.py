from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Generic, TypeVar
import logging

import numpy
import numpy.typing

from .data import DiffractionPatternArrayType, DiffractionPatternIndexes
from .object import ObjectArrayType
from .probe import ProbeArrayType
from .scan import ScanArrayType, ScanIndexes

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiffractionPatternStateData:
    indexes: DiffractionPatternIndexes
    array: DiffractionPatternArrayType


@dataclass(frozen=True)
class ScanStateData:
    indexes: ScanIndexes
    positionXInMeters: ScanArrayType
    positionYInMeters: ScanArrayType


@dataclass(frozen=True)
class ProbeStateData:
    pixelSizeXInMeters: float
    pixelSizeYInMeters: float
    array: ProbeArrayType


@dataclass(frozen=True)
class ObjectStateData:
    array: ObjectArrayType


@dataclass(frozen=True)
class StateData:
    # TODO include cost function values
    scan: ScanStateData
    probe: ProbeStateData
    object_: ObjectStateData
    diffractionPatterns: DiffractionPatternStateData | None = None


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
    PIXEL_SIZE_X: Final[str] = 'pixelSizeXInMeters'
    PIXEL_SIZE_Y: Final[str] = 'pixelSizeYInMeters'
    DATA_INDEXES: Final[str] = 'dataIndexes'
    DATA_ARRAY: Final[str] = 'data'
    SCAN_INDEXES: Final[str] = 'scanIndexes'
    SCAN_X: Final[str] = 'scanXInMeters'
    SCAN_Y: Final[str] = 'scanYInMeters'
    PROBE_ARRAY: Final[str] = 'probe'
    OBJECT_ARRAY: Final[str] = 'object'

    def __init__(self, dataCore: StatefulCore[DiffractionPatternStateData],
                 scanCore: StatefulCore[ScanStateData], probeCore: StatefulCore[ProbeStateData],
                 objectCore: StatefulCore[ObjectStateData]) -> None:
        self._dataCore = dataCore
        self._scanCore = scanCore
        self._probeCore = probeCore
        self._objectCore = objectCore

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

        try:
            scanState = ScanStateData(
                indexes=stateData[self.SCAN_INDEXES],
                positionXInMeters=stateData[self.SCAN_X],
                positionYInMeters=stateData[self.SCAN_Y],
            )
        except KeyError:
            logger.error('Failed to restore scan state.')
        else:
            self._scanCore.setStateData(scanState, filePath)

        try:
            probeState = ProbeStateData(pixelSizeXInMeters=float(stateData[self.PIXEL_SIZE_X]),
                                        pixelSizeYInMeters=float(stateData[self.PIXEL_SIZE_Y]),
                                        array=stateData[self.PROBE_ARRAY])
        except KeyError:
            logger.error('Failed to restore probe state.')
        else:
            self._probeCore.setStateData(probeState, filePath)

        try:
            objectState = ObjectStateData(array=stateData[self.OBJECT_ARRAY])
        except KeyError:
            logger.error('Failed to restore object state.')
        else:
            self._objectCore.setStateData(objectState, filePath)

    def saveStateData(self, filePath: Path, *, restartable: bool) -> None:
        '''save state data to file'''
        logger.debug(f'Writing state data to \"{filePath}\" [restartable={restartable}]')
        stateData: dict[str, numpy.typing.NDArray[Any]] = dict()

        if restartable:
            dataState = self._dataCore.getStateData()
            stateData[self.DATA_INDEXES] = dataState.indexes
            stateData[self.DATA_ARRAY] = dataState.array

        scanState = self._scanCore.getStateData()
        stateData[self.SCAN_INDEXES] = scanState.indexes
        stateData[self.SCAN_X] = scanState.positionXInMeters
        stateData[self.SCAN_Y] = scanState.positionYInMeters

        probeState = self._probeCore.getStateData()
        stateData[self.PIXEL_SIZE_X] = numpy.array(probeState.pixelSizeXInMeters)
        stateData[self.PIXEL_SIZE_Y] = numpy.array(probeState.pixelSizeYInMeters)
        stateData[self.PROBE_ARRAY] = probeState.array

        objectState = self._objectCore.getStateData()
        stateData[self.OBJECT_ARRAY] = objectState.array

        numpy.savez(filePath, **stateData)
