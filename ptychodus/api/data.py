from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from enum import Enum, auto
from pathlib import Path

import numpy

from .tree import SimpleTreeNode

ImageArrayType = numpy.typing.NDArray[numpy.integers]


class DatasetState(Enum):
    MISSING = auto()
    FOUND = auto()
    VALID = auto()


class DiffractionDataset(Sequence[ImageArrayType]):
    @abstractmethod
    def getState(self) -> DatasetState:
        pass

    @abstractmethod
    def getArray(self) -> ImageArrayType:
        pass


class DataFile(Sequence[DiffractionDataset]):
    pass


class DataFileReader(ABC):
    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def getFileContentsTree(self) -> SimpleTreeNode:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> DataFile:
        pass
