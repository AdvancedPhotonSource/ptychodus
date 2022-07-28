from __future__ import annotations
from pathlib import Path
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.probe import ProbeArrayType, ProbeFileReader
from .settings import ProbeSettings
from .sizer import ProbeSizer

logger = logging.getLogger(__name__)


class FileProbeInitializer(Observer):

    def __init__(self, settings: ProbeSettings, sizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ProbeFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._array = numpy.zeros(sizer.getProbeExtent().shape, dtype=complex)

    @classmethod
    def createInstance(cls, settings: ProbeSettings, sizer: ProbeSizer,
                       fileReaderChooser: PluginChooser[ProbeFileReader]) -> FileProbeInitializer:
        initializer = cls(settings, sizer, fileReaderChooser)

        settings.inputFileType.addObserver(initializer)
        initializer._fileReaderChooser.addObserver(initializer)
        initializer._syncFileReaderFromSettings()

        settings.inputFilePath.addObserver(initializer)
        initializer._openProbeFromSettings()

        return initializer

    def __call__(self) -> ProbeArrayType:
        return self._array

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openProbe(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            self._array = fileReader.read(filePath)
        else:
            logger.debug(f'Refusing to read invalid file path {filePath}')

    def openProbe(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.inputFilePath.value == filePath:
            self._openProbe(filePath)

        self._settings.inputFilePath.value = filePath

    def _openProbeFromSettings(self) -> None:
        self._openProbe(self._settings.inputFilePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.inputFileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.inputFilePath:
            self._openProbeFromSettings()
