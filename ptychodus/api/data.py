from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

from .tree import SimpleTreeNode


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
