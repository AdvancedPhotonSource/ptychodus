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

DiffractionArrayType = numpy.typing.NDArray[numpy.integer]


class DiffractionDataState(Enum):
    MISSING = auto()
    FOUND = auto()
    LOADED = auto()


class DiffractionData(Observable):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractmethod
    def getState(self) -> DiffractionDataState:
        pass

    @abstractmethod
    def getStartIndex(self) -> int:
        pass

    @abstractmethod
    def getArray(self) -> DiffractionArrayType:
        pass


@dataclass(frozen=True)
class DiffractionMetadata:
    filePath: Path
    imageWidth: int
    imageHeight: int
    totalNumberOfImages: int
    detectorNumberOfPixelsX: Optional[int] = None
    detectorNumberOfPixelsY: Optional[int] = None
    detectorPixelSizeXInMeters: Optional[Decimal] = None
    detectorPixelSizeYInMeters: Optional[Decimal] = None
    detectorDistanceInMeters: Optional[Decimal] = None
    cropCenterXInPixels: Optional[int] = None
    cropCenterYInPixels: Optional[int] = None
    probeEnergyInElectronVolts: Optional[Decimal] = None


class DiffractionDataset(Sequence[DiffractionData], Observable):

    @abstractmethod
    def getMetadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def getContentsTree(self) -> SimpleTreeNode:
        pass


class SimpleDiffractionDataset(DiffractionDataset):

    def __init__(self, metadata: DiffractionMetadata, contentsTree: SimpleTreeNode,
                 dataList: list[DiffractionData]) -> None:
        self._metadata = metadata
        self._contentsTree = contentsTree
        self._dataList = dataList

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionData:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionData]:
        ...

    def __getitem__(
            self, index: Union[int,
                               slice]) -> Union[DiffractionData, Sequence[DiffractionData]]:
        return self._dataList[index]

    def __len__(self) -> int:
        return len(self._dataList)


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
