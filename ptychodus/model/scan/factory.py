from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader, ScanPoint, TabularScan
from .cartesian import CartesianScanRepositoryItem
from .lissajous import LissajousScanRepositoryItem
from .repositoryItem import ScanRepositoryItem
from .settings import ScanSettings
from .spiral import SpiralScanRepositoryItem
from .tabular import ScanFileInfo, TabularScanRepositoryItem
from .transformed import TransformedScanRepositoryItem

logger = logging.getLogger(__name__)


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser

    def createTabularItem(self, scan: Scan,
                          fileInfo: Optional[ScanFileInfo]) -> ScanRepositoryItem:
        item = TabularScanRepositoryItem(scan, fileInfo)
        return TransformedScanRepositoryItem(self._rng, item)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _readScan(self, filePath: Path) -> list[ScanRepositoryItem]:
        itemList: list[ScanRepositoryItem] = list()

        if filePath is not None and filePath.is_file():
            fileType = self._fileReaderChooser.getCurrentSimpleName()
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            reader = self._fileReaderChooser.getCurrentStrategy()
            scanSequence = reader.read(filePath)
            fileInfo = ScanFileInfo(fileType, filePath)

            for scan in scanSequence:
                item = self.createTabularItem(scan, fileInfo)
                itemList.append(item)
        else:
            logger.debug(f'Refusing to read invalid file path \"{filePath}\"')

        return itemList

    def openScan(self, filePath: Path, fileFilter: str) -> list[ScanRepositoryItem]:
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        return self._readScan(filePath)

    def openScanFromSettings(self) -> list[ScanRepositoryItem]:
        fileInfo = ScanFileInfo.createFromSettings(self._settings)
        self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
        return self._readScan(fileInfo.filePath)

    def getItemNameList(self) -> list[str]:
        return ['Lissajous', 'Raster', 'Snake', 'Spiral']

    def createItem(self, name: str) -> Optional[ScanRepositoryItem]:
        item: Optional[ScanRepositoryItem] = None
        nameLower = name.casefold()

        if nameLower == 'lissajous':
            item = LissajousScanRepositoryItem()
        elif nameLower == 'raster':
            item = CartesianScanRepositoryItem(snake=False)
        elif nameLower == 'snake':
            item = CartesianScanRepositoryItem(snake=True)
        elif nameLower == 'spiral':
            item = SpiralScanRepositoryItem()
        elif nameLower == 'tabular':
            item = TabularScanRepositoryItem(
                TabularScan('Tabular', {0: ScanPoint(Decimal(), Decimal())}), None)

        if item is not None:
            item = TransformedScanRepositoryItem(self._rng, item)
            item.syncFromSettings(self._settings)

        return item
