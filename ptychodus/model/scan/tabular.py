from __future__ import annotations
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

import numpy

from ...api.scan import Scan, ScanPoint, TabularScan
from .repository import ScanRepositoryItem
from .settings import ScanSettings


@dataclass(frozen=True)
class ScanFileInfo:
    fileType: str
    filePath: Path

    @classmethod
    def createNull(cls) -> ScanFileInfo:
        return cls(fileType='', filePath=Path())

    @classmethod
    def createFromSettings(cls, settings: ScanSettings) -> ScanFileInfo:
        return cls(
            fileType=settings.inputFileType.value,
            filePath=settings.inputFilePath.value,
        )


class TabularScanRepositoryItem(ScanRepositoryItem):
    NAME: Final[str] = 'Tabular'

    def __init__(self, scan: Scan, fileInfo: Optional[ScanFileInfo]) -> None:
        super().__init__()
        self._scan = scan
        self._fileInfo = fileInfo

    @classmethod
    def createEmpty(cls) -> TabularScanRepositoryItem:
        return cls(TabularScan.createEmpty(), None)

    @property
    def name(self) -> str:
        return self._scan.name

    @property
    def category(self) -> str:
        return self.NAME

    @property
    def variant(self) -> str:
        return 'FromMemory' if self._fileInfo is None else 'FromFile'

    @property
    def canActivate(self) -> bool:
        return (self._fileInfo is not None)

    def syncFromSettings(self, settings: ScanSettings) -> None:
        # NOTE do not sync file info from settings
        pass

    def syncToSettings(self, settings: ScanSettings) -> None:
        if self._fileInfo is None:
            raise ValueError('Missing file info.')
        else:
            settings.inputFileType.value = self._fileInfo.fileType
            settings.inputFilePath.value = self._fileInfo.filePath

    def __iter__(self) -> Iterator[int]:
        return iter(self._scan)

    def __getitem__(self, index: int) -> ScanPoint:
        return self._scan[index]

    def __len__(self) -> int:
        return len(self._scan)

    def getFileInfo(self) -> Optional[ScanFileInfo]:
        return self._fileInfo

    def setFileInfo(self, fileInfo: ScanFileInfo) -> None:
        self._fileInfo = fileInfo
        self.notifyObservers()
