from __future__ import annotations
from pathlib import Path
import logging

import numpy

from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser
from ..api.scan import (ScanDictionary, ScanFileReader, ScanInitializer, ScanPoint,
                        ScanPointSequence)

logger = logging.getLogger(__name__)


class FileScanInitializer(ScanInitializer, Observer):

    def __init__(self, settings: ScanSettings,
                 fileReaderChooser: PluginChooser[ScanFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._fileReaderChooser = fileReaderChooser
        self._scanDictionary: dict[str, ScanPointSequence] = dict()

    @classmethod
    def createInstance(cls, settings: ScanSettings,
                       fileReaderChooser: PluginChooser[ScanFileReader]) -> FileScanInitializer:
        initializer = cls(settings, fileReaderChooser)

        settings.inputFileType.addObserver(initializer)
        initializer._fileReaderChooser.addObserver(initializer)
        initializer._syncFileReaderFromSettings()

        settings.inputFilePath.addObserver(initializer)
        initializer._openScanFromSettings()

        return initializer

    @property
    def simpleName(self) -> str:
        return 'FromFile'

    @property
    def displayName(self) -> str:
        return 'From File'

    def __getitem__(self, key: str) -> ScanPointSequence:
        return self._scanDictionary[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._scanDictionary)

    def __len__(self) -> int:
        return len(self._scanDictionary)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openScan(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            scanDictionary = fileReader.read(filePath)
            self._scanDictionary = dict(scanDictionary)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.inputFilePath.value == filePath:
            self._openScan(filePath)

        self._settings.inputFilePath.value = filePath

    def _openScanFromSettings(self) -> None:
        self._openScan(self._settings.inputFilePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.inputFileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.inputFilePath:
            self._openScanFromSettings()
