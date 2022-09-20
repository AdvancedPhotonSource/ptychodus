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

from ...api.data import (DiffractionArray, DiffractionArrayState, DiffractionDataType,
                         DiffractionDataset, DiffractionFileReader, DiffractionFileWriter,
                         DiffractionMetadata, SimpleDiffractionArray)
from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry, SettingsGroup
from ...api.tree import SimpleTreeNode
from ..detector import Detector
from .active import ActiveDiffractionDataset
from .assembler import DiffractionDataAssembler
from .crop import CropPresenter, CropSettings, CropSizer
from .settings import DataSettings
from .watcher import DataDirectoryWatcher

logger = logging.getLogger(__name__)


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, settings: DataSettings, activeDiffractionDataset: ActiveDiffractionDataset,
                 fileReaderChooser: PluginChooser[DiffractionFileReader],
                 fileWriterChooser: PluginChooser[DiffractionFileWriter]) -> None:
        super().__init__()
        self._settings = settings
        self._activeDiffractionDataset = activeDiffractionDataset
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(
            cls, settings: DataSettings, activeDiffractionDataset: ActiveDiffractionDataset,
            fileReaderChooser: PluginChooser[DiffractionFileReader],
            fileWriterChooser: PluginChooser[DiffractionFileWriter]
    ) -> DiffractionDatasetPresenter:
        presenter = cls(settings, activeDiffractionDataset, fileReaderChooser, fileWriterChooser)
        settings.fileType.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()
        settings.filePath.addObserver(presenter)
        presenter._openDiffractionFileFromSettings()
        activeDiffractionDataset.addObserver(presenter)
        return presenter

    def getScratchDirectory(self) -> Path:
        return self._settings.scratchDirectory.value

    def setScratchDirectory(self, directory: Path) -> None:
        self._settings.scratchDirectory.value = directory

    def getContentsTree(self) -> SimpleTreeNode:
        return self._activeDiffractionDataset.getContentsTree()

    def getArrayLabel(self, index: int) -> str:
        return self._activeDiffractionDataset[index].getLabel()

    def getArrayState(self, index: int) -> DiffractionArrayState:
        return self._activeDiffractionDataset[index].getState()

    def getNumberOfArrays(self) -> int:
        return len(self._activeDiffractionDataset)

    def openArray(self, dataPath: str) -> Any:  # FIXME
        filePath = self._activeDiffractionDataset.getMetadata().filePath
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
            dataset = fileReader.read(filePath)
            self._activeDiffractionDataset.switchTo(dataset)
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
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._activeDiffractionDataset)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.fileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.filePath:
            self._openDiffractionFileFromSettings()
        elif observable is self._activeDiffractionDataset:
            self.notifyObservers()


class DiffractionArrayPresenter(Observable, Observer):

    def __init__(self, dataset: DiffractionDataset) -> None:
        super().__init__()
        self._dataset = dataset
        self._data: DiffractionArray = SimpleDiffractionArray.createNullInstance()
        self._dataIndex = 0

    @classmethod
    def createInstance(cls, dataset: DiffractionDataset) -> DiffractionArrayPresenter:
        presenter = cls(dataset)
        dataset.addObserver(presenter)
        return presenter

    def setCurrentArrayIndex(self, index: int) -> None:
        try:
            data = self._dataset[index]
        except IndexError:
            logger.exception('Invalid data index!')
            return

        self._data.removeObserver(self)
        self._data = data
        self._data.addObserver(self)
        self._dataIndex = index

        self.notifyObservers()

    def getCurrentArrayIndex(self) -> int:
        return self._dataIndex

    def getNumberOfImages(self) -> int:
        return len(self._data.getData())  # FIXME get from shape

    def getImage(self, index: int) -> DiffractionDataType:
        return self._data.getData()[index]

    def update(self, observable: Observable) -> None:
        if observable is self._dataset:
            self._data.removeObserver(self)
            self._data = SimpleDiffractionArray.createNullInstance()
            self._dataIndex = 0
            self.notifyObservers()
        elif observable is self._data:
            self.notifyObservers()


class DataCore:

    def __init__(self, settingsRegistry: SettingsRegistry, detector: Detector,
                 fileReaderChooser: PluginChooser[DiffractionFileReader],
                 fileWriterChooser: PluginChooser[DiffractionFileWriter]) -> None:
        self.cropSettings = CropSettings.createInstance(settingsRegistry)
        self.cropSizer = CropSizer.createInstance(self.cropSettings, detector)
        self.cropPresenter = CropPresenter.createInstance(self.cropSettings, self.cropSizer)

        self._settings = DataSettings.createInstance(settingsRegistry)
        self.assembler = DiffractionDataAssembler(self._settings, self.cropSizer)
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(
            self._settings, self.assembler)
        self.activeDataset = ActiveDiffractionDataset.createInstance(self.assembler)

        self.diffractionDatasetPresenter = DiffractionDatasetPresenter.createInstance(
            self._settings, self.activeDataset, fileReaderChooser, fileWriterChooser)
        self.diffractionArrayPresenter = DiffractionArrayPresenter.createInstance(
            self.activeDataset)

    def start(self) -> None:
        self._dataDirectoryWatcher.start()

    def stop(self) -> None:
        self._dataDirectoryWatcher.stop()
