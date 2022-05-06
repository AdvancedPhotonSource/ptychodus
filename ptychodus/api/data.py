from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Sequence
from enum import Enum, auto
from pathlib import Path

import numpy
import numpy.typing

from .observer import Observable
from .tree import SimpleTreeNode

DataArrayType = numpy.typing.NDArray[numpy.integer]


class DatasetState(Enum):
    NOT_FOUND = auto()
    EXISTS = auto()
    VALID = auto()


class DiffractionDataset(Sequence[DataArrayType], Observable):
    @abstractproperty
    def datasetName(self) -> str:
        pass

    @abstractproperty
    def datasetState(self) -> DatasetState:
        pass

    @abstractmethod
    def getArray(self) -> DataArrayType:
        pass


class DataFile(Sequence[DiffractionDataset]):
    @abstractmethod
    def getContentsTree(self) -> SimpleTreeNode:
        pass


class DataFileReader(ABC):
    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> DataFile:
        pass
