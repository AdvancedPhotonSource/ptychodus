from __future__ import annotations
from pathlib import Path
import logging

import numpy

from ...api.object import ObjectArrayType, ObjectFileReader
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from .initializer import ObjectInitializer
from .settings import ObjectSettings
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class FileObjectInitializer(ObjectInitializer, Observer):

    def __init__(self, sizer: ObjectSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader]) -> None:
        super().__init__()
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._filePath = Path.home()

    @classmethod
    def createInstance(
            cls, settings: ObjectSettings, sizer: ObjectSizer,
            fileReaderChooser: PluginChooser[ObjectFileReader]) -> FileObjectInitializer:
        initializer = cls(sizer, fileReaderChooser)
        initializer.syncFromSettings(settings)
        fileReaderChooser.addObserver(initializer)
        return initializer

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        self._fileReaderChooser.setFromSimpleName(settings.inputFileType.value)
        self._filePath = settings.inputFilePath.value
        super().syncFromSettings(settings)

    def syncToSettings(self, settings: ObjectSettings) -> None:
        settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()
        settings.inputFilePath.value = self._filePath
        super().syncToSettings(settings)

    @property
    def displayName(self) -> str:
        return 'Open File...'

    @property
    def simpleName(self) -> str:
        return 'FromFile'

    def __call__(self) -> ObjectArrayType:
        array = numpy.zeros(self._sizer.getObjectExtent().shape, dtype=complex)

        if self._filePath is None:
            logger.error(f'Path is None!')
        elif self._filePath.is_file():
            fileType = self._fileReaderChooser.getCurrentSimpleName()
            logger.debug(f'Reading \"{self._filePath}\" as \"{fileType}\"')

            fileReader = self._fileReaderChooser.getCurrentStrategy()
            array = fileReader.read(self._filePath)
        else:
            logger.error(f'Path \"{self._filePath}\" is not a file!')

        return array

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def setOpenFileFilter(self, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def setOpenFilePath(self, filePath: Path) -> None:
        if self._filePath != filePath:
            self._filePath = filePath
            self.notifyObservers()

    def getOpenFilePath(self) -> Path:
        return self._filePath

    def update(self, observable: Observable) -> None:
        if observable is self._fileReaderChooser:
            self.notifyObservers()
