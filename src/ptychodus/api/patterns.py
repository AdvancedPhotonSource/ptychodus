from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import overload, Any, TypeAlias

import numpy
import numpy.typing

from .geometry import ImageExtent, PixelGeometry
from .tree import SimpleTreeNode

BooleanArrayType: TypeAlias = numpy.typing.NDArray[numpy.bool_]
PatternDataType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
PatternIndexesType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]


@dataclass(frozen=True)
class CropCenter:
    positionXInPixels: int
    positionYInPixels: int


class PatternState(Enum):
    UNKNOWN = auto()
    LOADING = auto()
    LOADED = auto()


class DiffractionPatternArray:
    @abstractmethod
    def getLabel(self) -> str:
        pass

    @abstractmethod
    def getIndexes(self) -> PatternIndexesType:
        pass

    @abstractmethod
    def getData(self) -> PatternDataType:
        pass

    def getState(self) -> PatternState:
        return PatternState.UNKNOWN

    def getNumberOfPatterns(self) -> int:
        return self.getData().shape[0]


class SimpleDiffractionPatternArray(DiffractionPatternArray):
    def __init__(
        self,
        label: str,
        indexes: PatternIndexesType,
        data: PatternDataType,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._data = data

    @classmethod
    def createNullInstance(cls) -> SimpleDiffractionPatternArray:
        indexes = numpy.array([0])
        data = numpy.zeros((1, 1, 1), dtype=numpy.uint16)
        return cls('Null', indexes, data)

    def getLabel(self) -> str:
        return self._label

    def getIndexes(self) -> PatternIndexesType:
        return self._indexes

    def getData(self) -> PatternDataType:
        return self._data


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


class DiffractionDataset(Sequence[DiffractionPatternArray]):
    @abstractmethod
    def getMetadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def getContentsTree(self) -> SimpleTreeNode:
        pass


class SimpleDiffractionDataset(DiffractionDataset):
    def __init__(
        self,
        metadata: DiffractionMetadata,
        contentsTree: SimpleTreeNode,
        arrayList: list[DiffractionPatternArray],
    ) -> None:
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
    def __getitem__(self, index: int) -> DiffractionPatternArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DiffractionPatternArray | Sequence[DiffractionPatternArray]:
        return self._arrayList[index]

    def __len__(self) -> int:
        return len(self._arrayList)


class DiffractionFileReader(ABC):
    """interface for plugins that read diffraction files"""

    @abstractmethod
    def read(self, filePath: Path) -> DiffractionDataset:
        """reads a diffraction dataset from file"""
        pass


class DiffractionFileWriter(ABC):
    """interface for plugins that write diffraction files"""

    @abstractmethod
    def write(self, filePath: Path, dataset: DiffractionDataset) -> None:
        """writes a diffraction dataset to file"""
        pass
