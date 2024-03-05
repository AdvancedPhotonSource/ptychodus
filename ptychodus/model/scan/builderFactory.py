from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import logging

from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanFileReader, ScanFileWriter
from .builder import FromFileScanBuilder, ScanBuilder
from .cartesian import CartesianScanBuilder, CartesianScanVariant
from .concentric import ConcentricScanBuilder
from .lissajous import LissajousScanBuilder
from .spiral import SpiralScanBuilder

logger = logging.getLogger(__name__)


class ScanBuilderFactory(Iterable[str]):

    @classmethod
    def _createBuilders(cls) -> Mapping[str, Callable[[], ScanBuilder]]:
        builders: dict[str, Callable[[], ScanBuilder]] = {
            variant.getDisplayName():
            lambda variant=variant: CartesianScanBuilder(variant)  # type: ignore
            for variant in CartesianScanVariant
        }
        builders.update({
            'Concentric': lambda: ConcentricScanBuilder(),
            'Spiral': lambda: SpiralScanBuilder(),
            'Lissajous': lambda: LissajousScanBuilder(),
        })
        return builders

    def __init__(self, fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._builders = ScanBuilderFactory._createBuilders()

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ScanBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown scan builder \"{name}\"!') from exc

        return factory()

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
