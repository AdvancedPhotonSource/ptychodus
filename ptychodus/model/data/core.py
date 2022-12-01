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

from ...api.data import (DiffractionDataset, DiffractionFileReader, DiffractionMetadata,
                         DiffractionPatternArray, DiffractionPatternData, DiffractionPatternState,
                         SimpleDiffractionDataset, SimpleDiffractionPatternArray)
from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry, SettingsGroup
from ...api.tree import SimpleTreeNode
from ..detector import Detector
from ..statefulCore import StateDataType, StatefulCore
from .crop import CropSizer
from .dataset import ActiveDiffractionDataset
from .patterns import DiffractionPatternPresenter
from .settings import DiffractionDatasetSettings, DiffractionPatternSettings
from .watcher import DataDirectoryWatcher

logger = logging.getLogger(__name__)


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, settings: DiffractionDatasetSettings,
                 activeDiffractionDataset: ActiveDiffractionDataset,
                 fileReaderChooser: PluginChooser[DiffractionFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._activeDiffractionDataset = activeDiffractionDataset
        self._fileReaderChooser = fileReaderChooser

    @classmethod
    def createInstance(
            cls, settings: DiffractionDatasetSettings,
            activeDiffractionDataset: ActiveDiffractionDataset,
            fileReaderChooser: PluginChooser[DiffractionFileReader]
    ) -> DiffractionDatasetPresenter:
        presenter = cls(settings, activeDiffractionDataset, fileReaderChooser)
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

    @property
    def isReadyToAssemble(self) -> bool:
        return self._activeDiffractionDataset.isReadyToAssemble

    @property
    def isAssembled(self) -> bool:
        return (len(self._activeDiffractionDataset) > 0)

    def getContentsTree(self) -> SimpleTreeNode:
        return self._activeDiffractionDataset.getContentsTree()

    def getArrayLabel(self, index: int) -> str:
        return self._activeDiffractionDataset[index].getLabel()

    def getPatternState(self, index: int) -> DiffractionPatternState:
        return self._activeDiffractionDataset[index].getState()

    def getNumberOfArrays(self) -> int:
        return len(self._activeDiffractionDataset)

    def getDatasetLabel(self) -> str:
        filePath = self._activeDiffractionDataset.getMetadata().filePath
        return filePath.stem if filePath else 'Unknown'

    def openArray(self, dataPath: str) -> Any:  # TODO generalize for other file formats
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
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def saveDiffractionFile(self, filePath: Path) -> None:
        fileFilter = self.getSaveFileFilter()
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        array = self._activeDiffractionDataset.getAssembledData()
        numpy.save(filePath, array)

    def startProcessingDiffractionPatterns(self) -> None:
        self._activeDiffractionDataset.start()

    def stopProcessingDiffractionPatterns(self, finishAssembling: bool) -> None:
        self._activeDiffractionDataset.stop(finishAssembling)

    def initializeStreaming(self, metadata: DiffractionMetadata) -> None:
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()
        dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)
        self._activeDiffractionDataset.switchTo(dataset)

    def assemble(self, array: DiffractionPatternArray) -> None:
        self._activeDiffractionDataset.insertArray(array)

    def getAssemblyQueueSize(self) -> int:
        return self._activeDiffractionDataset.getAssemblyQueueSize()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.fileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.filePath:
            self._openDiffractionFileFromSettings()
        elif observable is self._activeDiffractionDataset:
            self.notifyObservers()


class ActiveDiffractionPatternPresenter(Observable, Observer):

    def __init__(self, dataset: DiffractionDataset) -> None:
        super().__init__()
        self._dataset = dataset
        self._array: DiffractionPatternArray = SimpleDiffractionPatternArray.createNullInstance()

    @classmethod
    def createInstance(cls, dataset: DiffractionDataset) -> ActiveDiffractionPatternPresenter:
        presenter = cls(dataset)
        dataset.addObserver(presenter)
        return presenter

    def setCurrentPatternIndex(self, index: int) -> None:
        try:
            data = self._dataset[index]
        except IndexError:
            logger.exception('Invalid data index!')
            return

        self._array.removeObserver(self)
        self._array = data
        self._array.addObserver(self)
        self.notifyObservers()

    def getCurrentPatternIndex(self) -> int:
        return self._array.getIndex()

    def getNumberOfImages(self) -> int:
        return self._array.getData().shape[0]

    def getImage(self, index: int) -> DiffractionPatternData:
        return self._array.getData()[index]

    def update(self, observable: Observable) -> None:
        if observable is self._dataset:
            self._array.removeObserver(self)
            self._array = SimpleDiffractionPatternArray.createNullInstance()
            self.notifyObservers()
        elif observable is self._array:
            self.notifyObservers()


class DataCore(StatefulCore):

    def __init__(self, settingsRegistry: SettingsRegistry, detector: Detector,
                 fileReaderChooser: PluginChooser[DiffractionFileReader]) -> None:
        self._datasetSettings = DiffractionDatasetSettings.createInstance(settingsRegistry)
        self.patternSettings = DiffractionPatternSettings.createInstance(settingsRegistry)

        self.cropSizer = CropSizer.createInstance(self.patternSettings, detector)
        self.patternPresenter = DiffractionPatternPresenter.createInstance(
            self.patternSettings, self.cropSizer)

        self.dataset = ActiveDiffractionDataset(self._datasetSettings, self.patternSettings,
                                                self.cropSizer)
        self._dataDirectoryWatcher = DataDirectoryWatcher.createInstance(
            self._datasetSettings, self.dataset)

        self.diffractionDatasetPresenter = DiffractionDatasetPresenter.createInstance(
            self._datasetSettings, self.dataset, fileReaderChooser)
        self.activePatternPresenter = ActiveDiffractionPatternPresenter.createInstance(
            self.dataset)

    def getStateData(self, *, restartable: bool) -> StateDataType:
        state = dict()

        if restartable:
            state['dataIndex'] = numpy.array(self.dataset.getAssembledIndexes())
            state['data'] = self.dataset.getAssembledData()

        return state

    def setStateData(self, state: StateDataType) -> None:
        # FIXME use dataIndex = state['dataIndex']
        try:
            data = state['data']
        except KeyError:
            logger.debug('Skipped restoring data array state.')
        else:
            self.dataset.setAssembledData(data)

    def start(self) -> None:
        self._dataDirectoryWatcher.start()

    def stop(self) -> None:
        self._dataDirectoryWatcher.stop()
        self.dataset.stop(finishAssembling=False)
