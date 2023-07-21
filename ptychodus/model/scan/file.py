from pathlib import Path
from typing import Final
import logging

from ...api.scan import Scan, ScanFileReader
from .repository import ScanInitializer
from .settings import ScanSettings

logger = logging.getLogger(__name__)


class FromFileScanInitializer(ScanInitializer):
    SIMPLE_NAME: Final[str] = 'FromFile'
    DISPLAY_NAME: Final[str] = 'Open File...'

    def __init__(self, filePath: Path, fileType: str, fileReader: ScanFileReader) -> None:
        super().__init__()
        self._filePath = filePath
        self._fileType = fileType
        self._fileReader = fileReader

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ScanSettings) -> None:
        # NOTE do not sync file info from settings
        pass

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.inputFileType.value = self._fileType
        settings.inputFilePath.value = self._filePath

    def __call__(self) -> Scan:
        logger.debug(f'Reading \"{self._filePath}\" as \"{self._fileType}\"')

        try:
            scan = self._fileReader.read(self._filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{self._filePath}\"') from exc

        return scan
