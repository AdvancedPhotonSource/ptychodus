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
    # TODO must be saved to disk to make active; can be made active iff fileInfo not None

    def __init__(self,
                 parameters: ScanInitializerParameters,
                 pointList: list[ScanPoint],
                 name: str,
                 fileInfo: Optional[ScanFileInfo] = None) -> None:
        super().__init__(parameters)
        self._pointList = pointList
        self._name = name
        self._fileInfo = fileInfo

    @classmethod
    def createFromSettings(cls, rng: numpy.random.Generator, settings: ScanSettings,
                           name: str) -> TabularScanInitializer:
        parameters = ScanInitializerParameters.createFromSettings(rng, settings)
        pointList: list[ScanPoint] = list()  # FIXME
        fileInfo = ScanFileInfo.createFromSettings(settings)
        return cls(parameters, pointList, name, fileInfo)

    def syncToSettings(self, settings: ScanSettings) -> None:
        if self._fileInfo is None:
            return

        settings.initializer.value = 'FromFile'
        self._fileInfo.syncToSettings(settings)
        super().syncToSettings(settings)

    @property
    def category(self) -> str:
        return 'Tabular'

    @property
    def name(self) -> str:
        return self._name

    def _getPoint(self, index: int) -> ScanPoint:
        return self._pointList[index]

    def __len__(self) -> int:
        return len(self._pointList)
