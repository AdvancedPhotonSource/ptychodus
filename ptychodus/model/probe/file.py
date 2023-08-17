from pathlib import Path
from typing import Final
import logging

from ...api.probe import ProbeArrayType, ProbeFileReader
from .repository import ProbeInitializer
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class FromFileProbeInitializer(ProbeInitializer):
    SIMPLE_NAME: Final[str] = 'FromFile'
    DISPLAY_NAME: Final[str] = 'Open File...'

    def __init__(self, filePath: Path, fileType: str, fileReader: ProbeFileReader) -> None:
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

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        # NOTE do not sync file info from settings
        pass

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.inputFileType.value = self._fileType
        settings.inputFilePath.value = self._filePath

    def __call__(self) -> ProbeArrayType:
        logger.debug(f'Reading \"{self._filePath}\" as \"{self._fileType}\"')

        try:
            array = self._fileReader.read(self._filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{self._filePath}\"') from exc

        return array
