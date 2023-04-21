from pathlib import Path
from typing import Final
import logging

from ...api.scan import Scan, ScanFileReader
from .repository import ScanInitializer
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class FromFileScanInitializer(ScanInitializer):
    NAME: Final[str] = 'Open File...'

    def __init__(self, filePath: Path, fileReader: ScanFileReader) -> None:
        super().__init__()
        self._filePath = filePath
        self._fileReader = fileReader

    @property
    def simpleName(self) -> str:
        return 'FromFile'

    @property
    def displayName(self) -> str:
        return self.NAME

    def syncFromSettings(self, settings: ScanSettings) -> None:
        # NOTE do not sync file info from settings
        pass

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.inputFileType.value = self._fileReader.simpleName
        settings.inputFilePath.value = self._filePath

    def __call__(self) -> Scan:
        fileType = self._fileReader.simpleName
        logger.debug(f'Reading \"{self._filePath}\" as \"{fileType}\"')

        try:
            scan = self._fileReader.read(self._filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{self._filePath}\"') from exc

        return scan
