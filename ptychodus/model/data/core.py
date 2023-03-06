from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from typing import Any
import logging

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
from .builder import ActiveDiffractionDatasetBuilder
from .dataset import ActiveDiffractionDataset
from .io import DiffractionDatasetInputOutputPresenter
from .patterns import DiffractionPatternPresenter
from .settings import DiffractionDatasetSettings, DiffractionPatternSettings
from .sizer import DiffractionPatternSizer

logger = logging.getLogger(__name__)


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, settings: DiffractionDatasetSettings,
                 dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset

    @classmethod
    def createInstance(cls, settings: DiffractionDatasetSettings,
                       dataset: ActiveDiffractionDataset) -> DiffractionDatasetPresenter:
        presenter = cls(settings, dataset)
        settings.addObserver(presenter)
        dataset.addObserver(presenter)
        return presenter

    def isMemmapEnabled(self) -> bool:
        return self._settings.memmapEnabled.value

    def setMemmapEnabled(self, value: bool) -> None:
        self._settings.memmapEnabled.value = value

    def getScratchDirectory(self) -> Path:
        return self._settings.scratchDirectory.value

    def setScratchDirectory(self, directory: Path) -> None:
        self._settings.scratchDirectory.value = directory

    def getNumberOfDataThreadsLimits(self) -> Interval[int]:
        return Interval[int](1, 64)

    def getNumberOfDataThreads(self) -> int:
        limits = self.getNumberOfDataThreadsLimits()
        return limits.clamp(self._settings.numberOfDataThreads.value)

    def setNumberOfDataThreads(self, number: int) -> None:
        self._settings.numberOfDataThreads.value = number

    @property
    def isAssembled(self) -> bool:
        return (len(self._dataset) > 0)

    def getContentsTree(self) -> SimpleTreeNode:
        return self._dataset.getContentsTree()

    def getArrayLabel(self, index: int) -> str:
        return self._dataset[index].getLabel()

    def getPatternState(self, index: int) -> DiffractionPatternState:
        return self._dataset[index].getState()

    def getNumberOfArrays(self) -> int:
        return len(self._dataset)

    def getDatasetLabel(self) -> str:
        filePath = self._dataset.getMetadata().filePath
        return filePath.stem if filePath else 'Unknown'

    def openArray(self, dataPath: str) -> Any:  # TODO generalize for other file formats
        filePath = self._dataset.getMetadata().filePath
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

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers
        elif observable is self._dataset:
            self.notifyObservers()


class ActiveDiffractionPatternPresenter(Observable, Observer):

    # FIXME data isn't autoloading correctly
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

        self.patternSizer = DiffractionPatternSizer.createInstance(self.patternSettings, detector)
        self.patternPresenter = DiffractionPatternPresenter.createInstance(
            self.patternSettings, self.patternSizer)

        self.dataset = ActiveDiffractionDataset(self._datasetSettings, self.patternSettings,
                                                self.patternSizer)
        self._builder = ActiveDiffractionDatasetBuilder(self._datasetSettings, self.dataset)

        self.datasetPresenter = DiffractionDatasetPresenter.createInstance(
            self._datasetSettings, self.dataset)
        self.datasetInputOutputPresenter = DiffractionDatasetInputOutputPresenter.createInstance(
            self._datasetSettings, self.dataset, self._builder, fileReaderChooser)
        self.activePatternPresenter = ActiveDiffractionPatternPresenter.createInstance(
            self.dataset)

    def loadDiffractionDataset(self, filePath: Path, fileFilter: str) -> None:
        self.datasetInputOutputPresenter.setOpenFileFilter(fileFilter)
        self.datasetInputOutputPresenter.openDiffractionFile(filePath)
        self.datasetInputOutputPresenter.startProcessingDiffractionPatterns()
        self.datasetInputOutputPresenter.stopProcessingDiffractionPatterns(finishAssembling=True)

    def getStateData(self, *, restartable: bool) -> StateDataType:
        state = dict()

        if restartable:
            state['dataIndexes'] = numpy.array(self.dataset.getAssembledIndexes())
            state['data'] = self.dataset.getAssembledData()

        return state

    def setStateData(self, state: StateDataType) -> None:
        try:
            dataIndexes = state['dataIndexes']
            data = state['data']
        except KeyError:
            logger.debug('Skipped restoring data array state.')
            return

        if self._builder.isAssembling:
            self._builder.stop(finishAssembling=False)

        self.dataset.setAssembledData(data, dataIndexes)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self._builder.stop(finishAssembling=False)
