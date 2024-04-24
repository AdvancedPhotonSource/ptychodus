from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class WorkflowProductAPI(ABC):

    @abstractmethod
    def openScan(self, filePath: Path, fileType: str) -> None:
        pass

    @abstractmethod
    def buildScan(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        pass

    @abstractmethod
    def openProbe(self, filePath: Path, fileType: str) -> None:
        pass

    @abstractmethod
    def buildProbe(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        pass

    @abstractmethod
    def openObject(self, filePath: Path, fileType: str) -> None:
        pass

    @abstractmethod
    def buildObject(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        pass

    @abstractmethod
    def reconstruct(self) -> None:
        pass

    @abstractmethod
    def saveProduct(self, filePath: Path, fileType: str) -> None:
        pass


class WorkflowAPI(ABC):

    @abstractmethod
    def openPatterns(self, filePath: Path, fileType: str) -> None:
        '''loads diffraction patterns from file'''
        pass

    @abstractmethod
    def createProduct(self, name: str) -> WorkflowProductAPI:
        '''creates a new product'''
        pass


class FileBasedWorkflow(ABC):

    @abstractmethod
    def getFilePattern(self) -> str:
        '''UNIX-style filename pattern. For rules see fnmatch from Python standard library.'''
        pass

    @abstractmethod
    def execute(self, api: WorkflowAPI, filePath: Path) -> None:
        '''uses workflow API to execute the workflow'''
        pass
