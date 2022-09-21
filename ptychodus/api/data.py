from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, auto
from pathlib import Path
from typing import overload, Optional, Union

import numpy
import numpy.typing

from .observer import Observable
from .tree import SimpleTreeNode

DiffractionDataType = numpy.typing.NDArray[numpy.integer]


class DiffractionArrayState(Enum):
    MISSING = auto()
    FOUND = auto()
    LOADED = auto()


class DiffractionArray(Observable):

    @abstractmethod
    def getLabel(self) -> str:
        pass

    @abstractmethod
    def getIndex(self) -> int:
        pass

    @abstractmethod
    def getData(self) -> DiffractionDataType:
        pass

    @abstractmethod
    def getState(self) -> DiffractionArrayState:
        pass


class SimpleDiffractionArray(DiffractionArray):

    def __init__(self, label: str, index: int, data: DiffractionDataType,
                 state: DiffractionArrayState) -> None:
        super().__init__()
        self._label = label
        self._index = index
        self._data = data
        self._state = state

    @classmethod
    def createNullInstance(cls) -> DiffractionArray:
        data = numpy.zeros((1, 0, 0), dtype=numpy.uint16)
        state = DiffractionArrayState.MISSING
        return cls('Null', 0, data, state)

    def getLabel(self) -> str:
        return self._label

    def getIndex(self) -> int:
        return self._index

    def getData(self) -> DiffractionDataType:
        return self._data

    def getState(self) -> DiffractionArrayState:
        return self._state

    def setState(self, state: DiffractionArrayState) -> None:
        self._state = state
        self.notifyObservers()


@dataclass(frozen=True)
class DiffractionMetadata:
    filePath: Path
    imageWidth: int
    imageHeight: int
    numberOfImagesPerArray: int
    numberOfImagesTotal: int
    detectorNumberOfPixelsX: Optional[int] = None
    detectorNumberOfPixelsY: Optional[int] = None
    detectorPixelSizeXInMeters: Optional[Decimal] = None
    detectorPixelSizeYInMeters: Optional[Decimal] = None
    detectorDistanceInMeters: Optional[Decimal] = None
    cropCenterXInPixels: Optional[int] = None
    cropCenterYInPixels: Optional[int] = None
    probeEnergyInElectronVolts: Optional[Decimal] = None

    @classmethod
    def createNullInstance(cls) -> DiffractionMetadata:
        return cls(Path('/dev/null'), 0, 0, 0, 0)


class DiffractionDataset(Sequence[DiffractionArray], Observable):

    @abstractmethod
    def getMetadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def getContentsTree(self) -> SimpleTreeNode:
        pass


class SimpleDiffractionDataset(DiffractionDataset):

    def __init__(self, metadata: DiffractionMetadata, contentsTree: SimpleTreeNode,
                 arrayList: list[DiffractionArray]) -> None:
        super().__init__()
        self._metadata = metadata
        self._contentsTree = contentsTree
        self._arrayList = arrayList

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArray]:
        ...

    def __getitem__(
            self, index: Union[int, slice]) -> Union[DiffractionArray, Sequence[DiffractionArray]]:
        return self._arrayList[index]

    def __len__(self) -> int:
        return len(self._arrayList)


class DiffractionFileReader(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> DiffractionDataset:
        pass


class DiffractionFileWriter(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        pass
