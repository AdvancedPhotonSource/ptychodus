from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.scan import Scan, ScanFileReader, ScanFileWriter

from .builder import FromFileScanBuilder, ScanBuilder
from .cartesian import CartesianScanBuilder, CartesianScanVariant
from .concentric import ConcentricScanBuilder
from .lissajous import LissajousScanBuilder
from .settings import ScanSettings
from .spiral import SpiralScanBuilder

logger = logging.getLogger(__name__)


class ScanBuilderFactory(Iterable[str]):
    def __init__(
        self,
        settings: ScanSettings,
        fileReaderChooser: PluginChooser[ScanFileReader],
        fileWriterChooser: PluginChooser[ScanFileWriter],
    ) -> None:
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._builders: dict[str, Callable[[], ScanBuilder]] = {
            variant.name.lower(): lambda variant=variant: CartesianScanBuilder(variant, settings)  # type: ignore
            for variant in CartesianScanVariant
        }
        self._builders.update(
            {
                'concentric': lambda: ConcentricScanBuilder(settings),
                'spiral': lambda: SpiralScanBuilder(settings),
                'lissajous': lambda: LissajousScanBuilder(settings),
            }
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ScanBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown scan builder "{name}"!') from exc

        return factory()

    def createDefault(self) -> ScanBuilder:
        return next(iter(self._builders.values()))()

    def createFromSettings(self) -> ScanBuilder:
        name = self._settings.builder.getValue()
        nameRepaired = name.casefold()

        if nameRepaired == 'from_file':
            return self.createScanFromFile(
                self._settings.filePath.getValue(),
                self._settings.fileType.getValue(),
            )

        return self.create(nameRepaired)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createScanFromFile(self, filePath: Path, fileType: str) -> ScanBuilder:
        self._fileReaderChooser.setCurrentPluginByName(fileType)
        fileType = self._fileReaderChooser.currentPlugin.simpleName
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileScanBuilder(self._settings, filePath, fileType, fileReader)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveScan(self, filePath: Path, fileType: str, scan: Scan) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileType)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        fileWriter = self._fileWriterChooser.currentPlugin.strategy
        fileWriter.write(filePath, scan)
