from __future__ import annotations
from pathlib import Path
import logging

import numpy

from ...api.data import (DiffractionFileReader, DiffractionMetadata, DiffractionPatternArray,
                         DiffractionPatternData, DiffractionPatternState, SimpleDiffractionDataset)
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.tree import SimpleTreeNode
from .builder import ActiveDiffractionDatasetBuilder
from .dataset import ActiveDiffractionDataset
from .settings import DiffractionDatasetSettings

logger = logging.getLogger(__name__)


class DiffractionDatasetInputOutputPresenter(Observable, Observer):

    def __init__(self, settings: DiffractionDatasetSettings, dataset: ActiveDiffractionDataset,
                 builder: ActiveDiffractionDatasetBuilder,
                 fileReaderChooser: PluginChooser[DiffractionFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset
        self._builder = builder
        self._fileReaderChooser = fileReaderChooser

    @classmethod
    def createInstance(
        cls, settings: DiffractionDatasetSettings, dataset: ActiveDiffractionDataset,
        builder: ActiveDiffractionDatasetBuilder,
        fileReaderChooser: PluginChooser[DiffractionFileReader]
    ) -> DiffractionDatasetInputOutputPresenter:
        presenter = cls(settings, dataset, builder, fileReaderChooser)
        settings.fileType.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        presenter._syncFileReaderFromSettings()
        settings.filePath.addObserver(presenter)
        presenter._openDiffractionFileFromSettings()
        dataset.addObserver(presenter)
        return presenter

    @property
    def isReadyToAssemble(self) -> bool:
        return (self._builder.getAssemblyQueueSize() > 0)

    def assemble(self, array: DiffractionPatternArray) -> None:
        self._builder.insertArray(array)

    def startProcessingDiffractionPatterns(self) -> None:
        self._builder.start()

    def stopProcessingDiffractionPatterns(self, finishAssembling: bool) -> None:
        self._builder.stop(finishAssembling)

    def getAssemblyQueueSize(self) -> int:
        return self._builder.getAssemblyQueueSize()

    def initializeStreaming(self, metadata: DiffractionMetadata) -> None:
        contentsTree = SimpleTreeNode.createRoot(['Name', 'Type', 'Details'])
        arrayList: list[DiffractionPatternArray] = list()
        dataset = SimpleDiffractionDataset(metadata, contentsTree, arrayList)
        self._builder.switchTo(dataset)

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def setOpenFileFilter(self, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.fileType.value)
        self.notifyObservers()

    def _syncFileReaderToSettings(self) -> None:
        self._settings.fileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openDiffractionFile(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            dataset = fileReader.read(filePath)
            self._builder.switchTo(dataset)
        else:
            logger.debug(f'Refusing to read invalid file path {filePath}')

    def openDiffractionFile(self, filePath: Path) -> None:
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
        # TODO saveDiffractionFile should share code with state data I/O
        fileFilter = self.getSaveFileFilter()
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        array = self._dataset.getAssembledData()
        numpy.save(filePath, array)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.fileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.filePath:
            self._openDiffractionFileFromSettings()
