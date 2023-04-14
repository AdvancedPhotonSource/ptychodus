from pathlib import Path
from typing import Final
import logging

from ...api.object import ObjectArrayType, ObjectFileReader
from .repository import ObjectInitializer
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class FromFileObjectInitializer(ObjectInitializer):
    NAME: Final[str] = 'Open File...'

    def __init__(self, filePath: Path, fileReader: ObjectFileReader) -> None:
        super().__init__()
        self._filePath = filePath
        self._fileReader = fileReader

    @property
    def simpleName(self) -> str:
        return 'FromFile'

    @property
    def displayName(self) -> str:
        return self.NAME

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        # NOTE do not sync file info from settings
        pass

    def syncToSettings(self, settings: ObjectSettings) -> None:
        settings.inputFileType.value = self._fileReader.simpleName
        settings.inputFilePath.value = self._filePath

    def __call__(self) -> ObjectArrayType:
        fileType = self._fileReader.simpleName
        logger.debug(f'Reading \"{self._filePath}\" as \"{fileType}\"')

        try:
            array = self._fileReader.read(self._filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{self._filePath}\"') from exc

        return array
