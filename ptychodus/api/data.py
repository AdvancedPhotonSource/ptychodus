from abc import ABC, abstractmethod, abstractproperty
from enum import Enum, auto
from pathlib import Path

from .tree import SimpleTreeNode


class DatasetState(Enum):
    MISSING = auto()
    FOUND = auto()
    VALID = auto()


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
    def read(self, filePath: Path) -> None:
        pass
