from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import overload, Any, TypeAlias

import numpy
import numpy.typing

from .observer import Observable
from .tree import SimpleTreeNode

DiffractionPatternArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
DiffractionPatternIndexes: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]


@dataclass(frozen=True)
class PixelGeometry:
    widthInMeters: float
    heightInMeters: float

    @classmethod
    def createNull(cls) -> PixelGeometry:
        return cls(0., 0.)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.widthInMeters}, {self.heightInMeters})'


@dataclass(frozen=True)
class ImageExtent:
    widthInPixels: int
    heightInPixels: int

    @property
    def size(self) -> int:
        '''returns the number of pixels in the image'''
        return self.widthInPixels * self.heightInPixels

    @property
    def shape(self) -> tuple[int, int]:
        '''returns the image shape (heightInPixels, widthInPixels) tuple'''
        return self.heightInPixels, self.widthInPixels

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageExtent):
            return (self.shape == other.shape)

        return False

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.widthInPixels}, {self.heightInPixels})'


@dataclass(frozen=True)
class CropCenter:
    positionXInPixels: int
    positionYInPixels: int


class DiffractionPatternState(Enum):
    UNKNOWN = auto()
    MISSING = auto()
    FOUND = auto()
    LOADED = auto()


class DiffractionPatternArray(Observable):

    @abstractmethod
    def getLabel(self) -> str:
        pass

    @abstractmethod
    def getIndex(self) -> int:
        pass

    @abstractmethod
    def getData(self) -> DiffractionPatternArrayType:
        pass

    def getNumberOfPatterns(self) -> int:
        return self.getData().shape[0]

    @abstractmethod
    def getState(self) -> DiffractionPatternState:
        pass


class SimpleDiffractionPatternArray(DiffractionPatternArray):

    def __init__(self, label: str, index: int, data: DiffractionPatternArrayType,
                 state: DiffractionPatternState) -> None:
        super().__init__()
        self._label = label
        self._index = index
        self._data = data
        self._state = state

    @classmethod
    def createNullInstance(cls) -> SimpleDiffractionPatternArray:
        data = numpy.zeros((1, 1, 1), dtype=numpy.uint16)
        state = DiffractionPatternState.MISSING
        return cls('Null', 0, data, state)

    def getLabel(self) -> str:
        return self._label

    def getIndex(self) -> int:
        return self._index

    def getData(self) -> DiffractionPatternArrayType:
        return self._data

    def getState(self) -> DiffractionPatternState:
        return self._state


@dataclass(frozen=True)
class DiffractionMetadata:
    numberOfPatternsPerArray: int
    numberOfPatternsTotal: int
    patternDataType: numpy.dtype[numpy.integer[Any]]
    detectorDistanceInMeters: float | None = None
    detectorExtent: ImageExtent | None = None
    detectorPixelGeometry: PixelGeometry | None = None
    detectorBitDepth: int | None = None
    cropCenter: CropCenter | None = None
    probeEnergyInElectronVolts: float | None = None
    filePath: Path | None = None

    @classmethod
    def createNullInstance(cls, filePath: Path | None = None) -> DiffractionMetadata:
        return cls(0, 0, numpy.dtype(numpy.ubyte), filePath=filePath)


class DiffractionDataset(Sequence[DiffractionPatternArray], Observable):

    @abstractmethod
    def getMetadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def getContentsTree(self) -> SimpleTreeNode:
        pass


class SimpleDiffractionDataset(DiffractionDataset):

    def __init__(self, metadata: DiffractionMetadata, contentsTree: SimpleTreeNode,
                 arrayList: list[DiffractionPatternArray]) -> None:
        super().__init__()
        self._metadata = metadata
        self._contentsTree = contentsTree
        self._arrayList = arrayList

    @classmethod
    def createNullInstance(cls, filePath: Path | None = None) -> SimpleDiffractionDataset:
        metadata = DiffractionMetadata.createNullInstance(filePath)
        contentsTree = SimpleTreeNode.createRoot(list())
        arrayList: list[DiffractionPatternArray] = list()
        return cls(metadata, contentsTree, arrayList)

    def getMetadata(self) -> DiffractionMetadata:
        return self._metadata

    def getContentsTree(self) -> SimpleTreeNode:
        return self._contentsTree

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]:
        ...

    def __getitem__(self, index: int |slice) -> \
            DiffractionPatternArray | Sequence[DiffractionPatternArray]:
        return self._arrayList[index]

    def __len__(self) -> int:
        return len(self._arrayList)


class DiffractionFileReader(ABC):
    '''interface for plugins that read diffraction files'''

    @abstractmethod
    def read(self, filePath: Path) -> DiffractionDataset:
        '''reads a diffraction dataset from file'''
        pass


class DiffractionFileWriter(ABC):
    '''interface for plugins that write diffraction files'''

    @abstractmethod
    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        '''writes a diffraction dataset to file'''
        pass
