from collections.abc import Callable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan
from .cartesian import RasterScanRepositoryItem, SnakeScanRepositoryItem
from .indexFilters import ScanIndexFilterFactory
from .itemRepository import ScanRepositoryItem
from .lissajous import LissajousScanRepositoryItem
from .settings import ScanSettings
from .spiral import SpiralScanRepositoryItem
from .tabular import ScanFileInfo, TabularScanRepositoryItem
from .transformed import TransformedScanRepositoryItem

logger = logging.getLogger(__name__)


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 indexFilterFactory: ScanIndexFilterFactory,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._indexFilterFactory = indexFilterFactory
        self._fileReaderChooser = fileReaderChooser
        self._variants: Mapping[str, Callable[[], ScanRepositoryItem]] = {
            LissajousScanRepositoryItem.NAME.casefold(): LissajousScanRepositoryItem,
            RasterScanRepositoryItem.NAME.casefold(): RasterScanRepositoryItem,
            SnakeScanRepositoryItem.NAME.casefold(): SnakeScanRepositoryItem,
            SpiralScanRepositoryItem.NAME.casefold(): SpiralScanRepositoryItem,
        }

    def _transformed(self, item: ScanRepositoryItem) -> ScanRepositoryItem:
        transformedItem = TransformedScanRepositoryItem(self._rng, item, self._indexFilterFactory)
        transformedItem.syncFromSettings(self._settings)
        return transformedItem

    def createTabularItem(self, scan: Scan,
                          fileInfo: Optional[ScanFileInfo]) -> ScanRepositoryItem:
        return self._transformed(TabularScanRepositoryItem(scan, fileInfo))

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _readScan(self, filePath: Path) -> list[ScanRepositoryItem]:
        itemList: list[ScanRepositoryItem] = list()

        if filePath is not None and filePath.is_file():
            fileType = self._fileReaderChooser.getCurrentSimpleName()
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            fileInfo = ScanFileInfo(fileType, filePath)
            reader = self._fileReaderChooser.getCurrentStrategy()

            try:
                scanSequence = reader.read(filePath)
            except ScanPointParseError as ex:
                logger.exception(f'Failed to read \"{filePath}\"')
            else:
                for scan in scanSequence:
                    item = self.createTabularItem(scan, fileInfo)
                    itemList.append(item)
        else:
            logger.debug(f'Refusing to read invalid file path \"{filePath}\"')

        return itemList

    def openScan(self, filePath: Path, fileFilter: str) -> list[ScanRepositoryItem]:
        self._fileReaderChooser.setFromDisplayName(fileFilter)
        return self._readScan(filePath)

    def getItemNameList(self) -> list[str]:
        return [name.title() for name in self._variants]

    def createItem(self, initializerName: str) -> list[ScanRepositoryItem]:
        itemList: list[ScanRepositoryItem] = list()

        if initializerName.casefold() == 'fromfile':
            fileInfo = ScanFileInfo.createFromSettings(self._settings)
            self._fileReaderChooser.setFromSimpleName(fileInfo.fileType)
            itemList.extend(self._readScan(fileInfo.filePath))
        else:
            try:
                itemFactory = self._variants[initializerName.casefold()]
            except KeyError:
                logger.error(f'Unknown scan initializer \"{initializerName}\"!')
            else:
                itemList.append(itemFactory())

        return [self._transformed(item) for item in itemList]
