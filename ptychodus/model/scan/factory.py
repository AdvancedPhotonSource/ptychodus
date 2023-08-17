from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, TypeAlias
import logging

import numpy

from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader
from .cartesian import CartesianScanInitializer
from .concentric import ConcentricScanInitializer
from .file import FromFileScanInitializer
from .lissajous import LissajousScanInitializer
from .repository import ScanRepositoryItem
from .settings import ScanSettings
from .spiral import SpiralScanInitializer

logger = logging.getLogger(__name__)

InitializerFactory: TypeAlias = Callable[[], Optional[ScanRepositoryItem]]


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        self._rng = rng
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser
        self._initializers = PluginChooser[InitializerFactory]()
        self._initializers.registerPlugin(
            self.createItemFromFile,
            simpleName=FromFileScanInitializer.SIMPLE_NAME,
            displayName=FromFileScanInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createRasterItem,
            simpleName='Raster',
        )
        self._initializers.registerPlugin(
            self.createSnakeItem,
            simpleName='Snake',
        )
        self._initializers.registerPlugin(
            self.createCenteredRasterItem,
            simpleName='CenteredRaster',
            displayName='Centered Raster',
        )
        self._initializers.registerPlugin(
            self.createCenteredSnakeItem,
            simpleName='CenteredSnake',
            displayName='Centered Snake',
        )
        self._initializers.registerPlugin(
            self.createConcentricItem,
            simpleName=ConcentricScanInitializer.SIMPLE_NAME,
            displayName=ConcentricScanInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createSpiralItem,
            simpleName=SpiralScanInitializer.SIMPLE_NAME,
            displayName=SpiralScanInitializer.DISPLAY_NAME,
        )
        self._initializers.registerPlugin(
            self.createLissajousItem,
            simpleName=LissajousScanInitializer.SIMPLE_NAME,
            displayName=LissajousScanInitializer.DISPLAY_NAME,
        )

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createFileInitializer(self, filePath: Path,
                              fileType: str) -> Optional[FromFileScanInitializer]:
        self._fileReaderChooser.setCurrentPluginByName(fileType)
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileScanInitializer(filePath, self._fileReaderChooser.currentPlugin.simpleName,
                                       fileReader)

    def openItemFromFile(self, filePath: Path, fileType: str) -> Optional[ScanRepositoryItem]:
        item: Optional[ScanRepositoryItem] = None

        if filePath.is_file():
            initializer = self.createFileInitializer(filePath, fileType)

            if initializer is None:
                logger.error('Refusing to create item without initializer!')
            else:
                item = ScanRepositoryItem(self._rng, filePath.stem)
                item.setInitializer(initializer)
        else:
            logger.debug(f'Refusing to create item with invalid file path \"{filePath}\"')

        return item

    def createItemFromScan(self,
                           nameHint: str,
                           scan: Scan,
                           *,
                           filePath: Optional[Path] = None,
                           fileType: str = '') -> ScanRepositoryItem:
        item = ScanRepositoryItem(self._rng, nameHint, scan)

        if filePath is not None:
            if filePath.is_file():
                initializer = self.createFileInitializer(filePath, fileType)

                if initializer is None:
                    logger.error('Refusing to add null initializer!')
                else:
                    item.setInitializer(initializer)
            else:
                logger.debug(f'Refusing to add initializer with invalid file path \"{filePath}\"')

        return item

    def createItemFromFile(self) -> Optional[ScanRepositoryItem]:
        filePath = self._settings.inputFilePath.value
        fileType = self._settings.inputFileType.value
        return self.openItemFromFile(filePath, fileType)

    def createRasterItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=False, centered=False)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createSnakeItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=True, centered=False)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createCenteredRasterItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=False, centered=True)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createCenteredSnakeItem(self) -> ScanRepositoryItem:
        initializer = CartesianScanInitializer(snake=True, centered=True)
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createConcentricItem(self) -> ScanRepositoryItem:
        initializer = ConcentricScanInitializer()
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createSpiralItem(self) -> ScanRepositoryItem:
        initializer = SpiralScanInitializer()
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def createLissajousItem(self) -> ScanRepositoryItem:
        initializer = LissajousScanInitializer()
        item = ScanRepositoryItem(self._rng, initializer.simpleName)
        item.setInitializer(initializer)
        return item

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return self._initializers.getDisplayNameList()

    def createItemFromInitializerName(self, name: str) -> Optional[ScanRepositoryItem]:
        item: Optional[ScanRepositoryItem] = None

        try:
            plugin = self._initializers[name]
        except KeyError:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            item = plugin.strategy()

        return item
