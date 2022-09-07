from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.scan import ScanFileReader, ScanPoint
from .cartesian import CartesianScanInitializer
from .initializer import ScanInitializer, ScanInitializerParameters
from .lissajous import LissajousScanInitializer
from .settings import ScanSettings
from .spiral import SpiralScanInitializer
from .tabular import ScanFileInfo, TabularScanInitializer

logger = logging.getLogger(__name__)


class ScanInitializerFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser

    def createInitializerParameters(self) -> ScanInitializerParameters:
        return ScanInitializerParameters.createFromSettings(self._rng, self._settings)

    def createTabularInitializer(self, pointList: list[ScanPoint], nameHint: str,
                                 fileInfo: Optional[ScanFileInfo]) -> TabularScanInitializer:
        parameters = self.createInitializerParameters()
        return TabularScanInitializer(parameters, pointList, nameHint, fileInfo)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _readScan(self, filePath: Path) -> list[TabularScanInitializer]:
        initializerList: list[TabularScanInitializer] = list()

        if filePath is not None and filePath.is_file():
            fileType = self._fileReaderChooser.getCurrentSimpleName()
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            reader = self._fileReaderChooser.getCurrentStrategy()
            namedSequenceDict = reader.read(filePath)
            fileInfo = ScanFileInfo(fileType, filePath)

            for name, pointSequence in namedSequenceDict.items():
                pointList = [point for point in pointSequence]
                initializer = self.createTabularInitializer(pointList, name, fileInfo)
                initializerList.append(initializer)
        else:
            logger.debug(f'Refusing to read invalid file path \"{filePath}\"')

        return initializerList

    def openScan(self, filePath: Path, fileFilter: str) -> list[TabularScanInitializer]:
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        return self._readScan(filePath)

    def openScanFromSettings(self) -> list[TabularScanInitializer]:
        fileInfo = ScanFileInfo.createFromSettings(self._settings)
        self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
        return self._readScan(fileInfo.filePath)

    def getInitializerNameList(self) -> list[str]:
        return ['Lissajous', 'Raster', 'Snake', 'Spiral']

    def createInitializer(self, name: str) -> Optional[ScanInitializer]:
        nameLower = name.casefold()
        parameters = self.createInitializerParameters()

        if nameLower == 'lissajous':
            return LissajousScanInitializer.createFromSettings(parameters, self._settings)
        elif nameLower == 'raster':
            return CartesianScanInitializer.createFromSettings(parameters,
                                                               self._settings,
                                                               snake=False)
        elif nameLower == 'snake':
            return CartesianScanInitializer.createFromSettings(parameters,
                                                               self._settings,
                                                               snake=True)
        elif nameLower == 'spiral':
            return SpiralScanInitializer.createFromSettings(parameters, self._settings)

        return None
