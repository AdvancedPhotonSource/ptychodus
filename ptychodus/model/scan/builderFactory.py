from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
import logging

from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader, ScanFileWriter
from .builder import FromFileScanBuilder, ScanBuilder
from .cartesian import CartesianScanBuilder
from .concentric import ConcentricScanBuilder
from .lissajous import LissajousScanBuilder
from .spiral import SpiralScanBuilder

logger = logging.getLogger(__name__)


class ScanBuilderFactory(Mapping[str, ScanBuilder]):

    def __init__(self, fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._builders: Mapping[str, Callable[[], ScanBuilder]] = {
            'Raster': lambda: CartesianScanBuilder(snake=False, centered=False),
            'Snake': lambda: CartesianScanBuilder(snake=True, centered=False),
            'Centered Raster': lambda: CartesianScanBuilder(snake=False, centered=True),
            'Centered Snake': lambda: CartesianScanBuilder(snake=True, centered=True),
            'Concentric': lambda: ConcentricScanBuilder(),
            'Spiral': lambda: SpiralScanBuilder(),
            'Lissajous': lambda: LissajousScanBuilder(),
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def __getitem__(self, name: str) -> ScanBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown scan builder \"{name}\"!') from exc

        return factory()

    def __len__(self) -> int:
        return len(self._builders)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createScanFromFile(self, filePath: Path, fileFilter: str) -> ScanBuilder:
        self._fileReaderChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileReaderChooser.currentPlugin.simpleName
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileScanBuilder(filePath, fileType, fileReader)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveScan(self, filePath: Path, fileFilter: str, scan: Scan) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        fileWriter = self._fileWriterChooser.currentPlugin.strategy
        fileWriter.write(filePath, scan)
