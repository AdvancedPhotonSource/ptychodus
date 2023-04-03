from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from .repository import ObjectRepositoryItem
from .settings import ObjectSettings


@dataclass(frozen=True)
class ObjectFileInfo:
    fileType: str
    filePath: Path

    @classmethod
    def createNull(cls) -> ObjectFileInfo:
        return cls(fileType='', filePath=Path())

    @classmethod
    def createFromSettings(cls, settings: ObjectSettings) -> ObjectFileInfo:
        return cls(
            fileType=settings.inputFileType.value,
            filePath=settings.inputFilePath.value,
        )


class SimpleObjectRepositoryItem(ObjectRepositoryItem):
    NAME: Final[str] = 'Simple'

    def __init__(self, name: str, array: ObjectArrayType,
                 fileInfo: Optional[ObjectFileInfo]) -> None:
        super().__init__()
        self._name = name
        self._array = array
        self._fileInfo = fileInfo

    @property
    def nameHint(self) -> str:
        return self._name

    @property
    def initializer(self) -> str:
        return 'FromMemory' if self._fileInfo is None else 'FromFile'

    @property
    def canSelect(self) -> bool:
        return (self._fileInfo is not None)

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        # NOTE do not sync file info from settings
        pass

    def syncToSettings(self, settings: ObjectSettings) -> None:
        if self._fileInfo is None:
            raise ValueError('Missing file info.')
        else:
            settings.inputFileType.value = self._fileInfo.fileType
            settings.inputFilePath.value = self._fileInfo.filePath

    def getDataType(self) -> str:
        return str(self._array.dtype)

    def getExtentInPixels(self) -> ImageExtent:
        return ImageExtent(width=self._array.shape[1], height=self._array.shape[0])

    def getSizeInBytes(self) -> int:
        return self._array.nbytes

    def getArray(self) -> ObjectArrayType:
        return self._array

    def getFileInfo(self) -> Optional[ObjectFileInfo]:
        return self._fileInfo

    def setFileInfo(self, fileInfo: ObjectFileInfo) -> None:
        self._fileInfo = fileInfo
        self.notifyObservers()
