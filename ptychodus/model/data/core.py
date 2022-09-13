from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import overload, Any, Optional, Union
import concurrent.futures
import functools
import logging
import tempfile

import h5py
import numpy

from ...api.data import (DiffractionArrayType, DiffractionData, DiffractionDataState,
                         DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                         DiffractionMetadata)
from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry, SettingsGroup
from ...api.tree import SimpleTreeNode

from .crop import CropSizer
from .settings import DataSettings
from .watcher import DataDirectoryWatcher

logger = logging.getLogger(__name__)


class DiffractionFilePresenter(Observable, Observer):

    def __init__(self, settings: DataSettings, activeDiffractionFile: ActiveDiffractionFile,
                 fileReaderChooser: PluginChooser[DiffractionFileReader],
                 fileWriterChooser: PluginChooser[DiffractionFileWriter]) -> None:
        super().__init__()
        self._settings = settings
        self._activeDiffractionFile = activeDiffractionFile
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(
            cls, settings: DataSettings, activeDiffractionFile: ActiveDiffractionFile,
            fileReaderChooser: PluginChooser[DiffractionFileReader],
            fileWriterChooser: PluginChooser[DiffractionFileWriter]) -> DiffractionFilePresenter:
        presenter = cls(settings, activeDiffractionFile, fileReaderChooser, fileWriterChooser)
        settings.fileType.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()
        settings.filePath.addObserver(presenter)
        presenter._openDiffractionFileFromSettings()
        activeDiffractionFile.addObserver(presenter)
        return presenter

    def getScratchDirectory(self) -> Path:
        return self._settings.scratchDirectory.value

    def setScratchDirectory(self, directory: Path) -> None:
        self._settings.scratchDirectory.value = directory

    def getContentsTree(self) -> SimpleTreeNode:
        return self._activeDiffractionFile.getContentsTree()

    def getDatasetName(self, index: int) -> str:
        return self._activeDiffractionFile[index].datasetName

    def getDiffractionDataState(self, index: int) -> DiffractionDataState:
        return self._activeDiffractionFile[index].datasetState

    def getNumberOfDatasets(self) -> int:
        return len(self._activeDiffractionFile)

    def openDataset(self, dataPath: str) -> Any:
        filePath = self._activeDiffractionFile.metadata.filePath
        data = None

        if filePath and h5py.is_hdf5(filePath) and dataPath:
            try:
                with h5py.File(filePath, 'r') as h5File:
                    if dataPath in h5File:
                        item = h5File.get(dataPath)

                        if isinstance(item, h5py.Dataset):
                            data = item[()]  # TODO decode strings as needed
                    else:
                        parentPath, attrName = dataPath.rsplit('/', 1)

                        if parentPath in h5File:
                            item = h5File.get(parentPath)

                            if attrName in item.attrs:
                                attr = item.attrs[attrName]
                                stringInfo = h5py.check_string_dtype(attr.dtype)

                                if stringInfo:
                                    data = attr.decode(stringInfo.encoding)
                                else:
                                    data = attr
            except OSError:
                logger.exception('Failed to open dataset!')

        return data

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.fileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.fileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openDiffractionFile(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            dataFile = fileReader.read(filePath)
            self._activeDiffractionFile.setActive(dataFile)
        else:
            logger.debug(f'Refusing to read invalid file path {filePath}')

    def openDiffractionFile(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.filePath.value == filePath:
            self._openDiffractionFile(filePath)

        self._settings.filePath.value = filePath

    def _openDiffractionFileFromSettings(self) -> None:
        self._openDiffractionFile(self._settings.filePath.value)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveDiffractionFile(self, filePath: Path, fileFilter: str) -> None:
        # TODO only save populated frames
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._activeDiffractionFile.getDiffractionData(cropped=False))

    def update(self, observable: Observable) -> None:
        if observable is self._settings.fileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.filePath:
            self._openDiffractionFileFromSettings()
        elif observable is self._activeDiffractionFile:
            self.notifyObservers()


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, dataFile: ActiveDiffractionFile) -> None:
        super().__init__()
        self._dataFile = dataFile
        self._dataset: DiffractionDataset = NullDiffractionDataset()
        self._datasetIndex = 0

    @classmethod
    def createInstance(cls, dataFile: ActiveDiffractionFile) -> DiffractionDatasetPresenter:
        presenter = cls(dataFile)
        dataFile.addObserver(presenter)
        return presenter

    def setCurrentDatasetIndex(self, index: int) -> None:
        try:
            dataset = self._dataFile[index]
        except IndexError:
            logger.exception('Invalid dataset index!')
            return

        self._dataset.removeObserver(self)
        self._dataset = dataset
        self._dataset.addObserver(self)
        self._datasetIndex = index

        self.notifyObservers()

    def getCurrentDatasetIndex(self) -> int:
        return self._datasetIndex

    def getNumberOfImages(self) -> int:
        return len(self._dataset)

    def getImage(self, index: int) -> DiffractionArrayType:
        return self._dataset[index]

    def update(self, observable: Observable) -> None:
        if observable is self._dataFile:
            self._dataset.removeObserver(self)
            self._dataset = NullDiffractionDataset()
            self._datasetIndex = 0
            self.notifyObservers()
        elif observable is self._dataset:
            self.notifyObservers()


class DataCore:

    def __init__(self, detector: Detector, fileReaderChooser: PluginChooser[DiffractionFileReader],
                 fileWriterChooser: PluginChooser[DiffractionFileWriter]) -> None:
        self.cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self.cropSizer = CropSizer.createInstance(self.cropSettings, detector)

        self._dataSettings = DataSettings.createInstance(self.settingsRegistry)
        self._activeDiffractionFile = ActiveDiffractionFile(self._dataSettings, self.cropSizer)

        self.cropPresenter = CropPresenter.createInstance(self.cropSettings, self.cropSizer)

        # TODO DataDirectoryWatcher should be optional
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(self._dataSettings)

        self.dataFilePresenter = DiffractionFilePresenter.createInstance(
            self._dataSettings, self._activeDiffractionFile,
            self._pluginRegistry.buildDiffractionFileReaderChooser(),
            self._pluginRegistry.buildDiffractionFileWriterChooser())
        self.diffractionDatasetPresenter = DiffractionDatasetPresenter.createInstance(
            self._activeDiffractionFile)

    def start(self) -> None:
        if self._dataDirectoryWatcher:
            self._dataDirectoryWatcher.start()

    def stop(self) -> None:
        if self._dataDirectoryWatcher:
            self._dataDirectoryWatcher.stop()
