from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import CropCenter
from ptychodus.api.settings import PathPrefixChange


class WorkflowProductAPI(ABC):

    @abstractmethod
    def openScan(self, filePath: Path, *, fileType: str | None = None) -> None:
        pass

    @abstractmethod
    def buildScan(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        pass

    @abstractmethod
    def openProbe(self, filePath: Path, *, fileType: str | None = None) -> None:
        pass

    @abstractmethod
    def buildProbe(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        pass

    @abstractmethod
    def buildProbeFromSettings(self) -> None:
        pass

    @abstractmethod
    def openObject(self, filePath: Path, *, fileType: str | None = None) -> None:
        pass

    @abstractmethod
    def buildObject(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        pass

    @abstractmethod
    def buildObjectFromSettings(self) -> None:
        pass

    @abstractmethod
    def reconstructLocal(self) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def reconstructRemote(self) -> None:
        pass

    @abstractmethod
    def saveProduct(self, filePath: Path, *, fileType: str | None = None) -> None:
        pass


class WorkflowAPI(ABC):

    @abstractmethod
    def openPatterns(
        self,
        filePath: Path,
        *,
        fileType: str | None = None,
        cropCenter: CropCenter | None = None,
        cropExtent: ImageExtent | None = None,
    ) -> None:
        '''opens diffraction patterns from file'''
        pass

    @abstractmethod
    def importProcessedPatterns(self, filePath: Path) -> None:
        '''import processed patterns'''
        pass

    @abstractmethod
    def exportProcessedPatterns(self, filePath: Path) -> None:
        '''export processed patterns'''
        pass

    @abstractmethod
    def openProduct(self, filePath: Path, *, fileType: str | None = None) -> WorkflowProductAPI:
        '''opens product from file'''
        pass

    @abstractmethod
    def createProduct(
        self,
        name: str,
        *,
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonsPerSecond: float | None = None,
        exposureTimeInSeconds: float | None = None,
    ) -> WorkflowProductAPI:
        '''creates a new product'''
        pass

    @abstractmethod
    def saveSettings(self,
                     filePath: Path,
                     changePathPrefix: PathPrefixChange | None = None) -> None:
        pass


class FileBasedWorkflow(ABC):

    @property
    @abstractmethod
    def isWatchRecursive(self) -> bool:
        '''indicates whether the data directory must be watched recursively'''
        pass

    @abstractmethod
    def getWatchFilePattern(self) -> str:
        '''UNIX-style filename pattern. For rules see fnmatch from Python standard library.'''
        pass

    @abstractmethod
    def execute(self, api: WorkflowAPI, filePath: Path) -> None:
        '''uses workflow API to execute the workflow'''
        pass
