from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


@dataclass
class ScanFileInfo:
    fileType: str
    filePath: Path
    fileSeriesKey: str

    @classmethod
    def createFromSettings(cls, settings: ScanSettings) -> ScanFileInfo:
        fileType = settings.inputFileType.value
        filePath = settings.inputFilePath.value
        fileSeriesKey = settings.inputFileSeriesKey.value
        return cls(fileType, filePath, fileSeriesKey)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.inputFileType.value = self.fileType
        settings.inputFilePath.value = self.filePath
        settings.inputFileSeriesKey.value = self.fileSeriesKey


class TabularScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters, pointList: list[ScanPoint],
                 fileInfo: Optional[ScanFileInfo]) -> None:
        super().__init__(parameters)
        self._pointList = pointList
        self._fileInfo = fileInfo

    def syncToSettings(self, settings: ScanSettings) -> None:
        if self._fileInfo is None:
            raise ValueError('Missing file info.')

        settings.initializer.value = self.variant
        self._fileInfo.syncToSettings(settings)
        super().syncToSettings(settings)

    @property
    def category(self) -> str:
        return 'Tabular'

    @property
    def variant(self) -> str:
        return 'FromMemory' if self._fileInfo is None else 'FromFile'

    def getFileInfo(self) -> Optional[ScanFileInfo]:
        return self._fileInfo

    def setFileInfo(self, fileInfo: ScanFileInfo) -> None:
        self._fileInfo = fileInfo
        self.notifyObservers()

    def _getPoint(self, index: int) -> ScanPoint:
        return self._pointList[index]

    def __len__(self) -> int:
        return len(self._pointList)
