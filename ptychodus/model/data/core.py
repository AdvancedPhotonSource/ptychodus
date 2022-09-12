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
from .watcher import DataDirectoryWatcher

logger = logging.getLogger(__name__)


class CachedDiffractionDataset(DiffractionDataset, Observer):

    def __init__(self, name: str, state: DiffractionDataState, array: DiffractionArrayType,
                 cropSizer: CropSizer) -> None:
        super().__init__()
        self._datasetName = name
        self._datasetState = state
        self._array = array
        self._cropSizer = cropSizer

    @classmethod
    def createInstance(cls, name: str, state: DiffractionDataState, array: DiffractionArrayType,
                       cropSizer: CropSizer) -> CachedDiffractionDataset:
        dataset = cls(name, state, array, cropSizer)
        cropSizer.addObserver(dataset)
        return dataset

    @property
    def datasetName(self) -> str:
        return self._datasetName

    @property
    def datasetState(self) -> DiffractionDataState:
        return self._datasetState

    def getArray(self) -> DiffractionArrayType:
        if self._cropSizer.isCropEnabled():
            sliceX = self._cropSizer.getSliceX()
            sliceY = self._cropSizer.getSliceY()
            return self._array[:, sliceY, sliceX]

        return self._array

    def __getitem__(self, index: int) -> DiffractionArrayType:
        array = numpy.empty((0, 0), dtype=numpy.uint16)

        if self._cropSizer.isCropEnabled():
            sliceX = self._cropSizer.getSliceX()
            sliceY = self._cropSizer.getSliceY()

            array = self._array[index, sliceY, sliceX]
        else:
            array = self._array[index, :, :]

        return array

    def __len__(self) -> int:
        return self._array.shape[0]

    def update(self, observable: Observable) -> None:
        if observable is self._cropSizer:
            self.notifyObservers()


class DataFilePresenter(Observable, Observer):

    def __init__(self, settings: DataSettings, activeDataFile: ActiveDataFile,
                 fileReaderChooser: PluginChooser[DataFileReader],
                 fileWriterChooser: PluginChooser[DataFileWriter]) -> None:
        super().__init__()
        self._settings = settings
        self._activeDataFile = activeDataFile
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, settings: DataSettings, activeDataFile: ActiveDataFile,
                       fileReaderChooser: PluginChooser[DataFileReader],
                       fileWriterChooser: PluginChooser[DataFileWriter]) -> DataFilePresenter:
        presenter = cls(settings, activeDataFile, fileReaderChooser, fileWriterChooser)
        settings.fileType.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()
        settings.filePath.addObserver(presenter)
        presenter._openDataFileFromSettings()
        activeDataFile.addObserver(presenter)
        return presenter

    def getScratchDirectory(self) -> Path:
        return self._settings.scratchDirectory.value

    def setScratchDirectory(self, directory: Path) -> None:
        self._settings.scratchDirectory.value = directory

    def getContentsTree(self) -> SimpleTreeNode:
        return self._activeDataFile.getContentsTree()

    def getDatasetName(self, index: int) -> str:
        return self._activeDataFile[index].datasetName

    def getDiffractionDataState(self, index: int) -> DiffractionDataState:
        return self._activeDataFile[index].datasetState

    def getNumberOfDatasets(self) -> int:
        return len(self._activeDataFile)

    def openDataset(self, dataPath: str) -> Any:
        filePath = self._activeDataFile.metadata.filePath
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

    def _openDataFile(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            dataFile = fileReader.read(filePath)
            self._activeDataFile.setActive(dataFile)
        else:
            logger.debug(f'Refusing to read invalid file path {filePath}')

    def openDataFile(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.filePath.value == filePath:
            self._openDataFile(filePath)

        self._settings.filePath.value = filePath

    def _openDataFileFromSettings(self) -> None:
        self._openDataFile(self._settings.filePath.value)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveDataFile(self, filePath: Path, fileFilter: str) -> None:
        # TODO only save populated frames
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._activeDataFile.getDiffractionData(cropped=False))

    def update(self, observable: Observable) -> None:
        if observable is self._settings.fileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.filePath:
            self._openDataFileFromSettings()
        elif observable is self._activeDataFile:
            self.notifyObservers()


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, dataFile: ActiveDataFile) -> None:
        super().__init__()
        self._dataFile = dataFile
        self._dataset: DiffractionDataset = NullDiffractionDataset()
        self._datasetIndex = 0

    @classmethod
    def createInstance(cls, dataFile: ActiveDataFile) -> DiffractionDatasetPresenter:
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
        self._cropSettings = CropSettings.createInstance(self.settingsRegistry)
        self._cropSizer = CropSizer.createInstance(self._cropSettings, detector)

        self._dataSettings = DataSettings.createInstance(self.settingsRegistry)
        self._activeDataFile = ActiveDataFile(self._dataSettings, self._cropSizer)

        self.cropPresenter = CropPresenter.createInstance(self._cropSettings, self._cropSizer)

        # TODO DataDirectoryWatcher should be optional
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(self._dataSettings)

        self.dataFilePresenter = DataFilePresenter.createInstance(
            self._dataSettings, self._activeDataFile,
            self._pluginRegistry.buildDataFileReaderChooser(),
            self._pluginRegistry.buildDataFileWriterChooser())
        self.diffractionDatasetPresenter = DiffractionDatasetPresenter.createInstance(
            self._activeDataFile)

    def start(self) -> None:
        if self._dataDirectoryWatcher:
            self._dataDirectoryWatcher.start()

    def stop(self) -> None:
        if self._dataDirectoryWatcher:
            self._dataDirectoryWatcher.stop()
