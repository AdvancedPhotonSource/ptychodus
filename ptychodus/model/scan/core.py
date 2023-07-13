from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import ScanFileReader, ScanFileWriter, ScanPoint, TabularScan
from ...api.settings import SettingsRegistry
from ...api.state import ScanStateData, StatefulCore
from ..data import ActiveDiffractionDataset
from .api import ScanAPI
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository, ScanRepositoryItem
from .selected import ScanRepositoryItemSettingsDelegate, SelectedScan
from .settings import ScanSettings
from .sizer import ScanSizer
from .streaming import StreamingScanBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanRepositoryItemPresenter:
    name: str
    item: ScanRepositoryItem


class ScanRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ScanRepository, itemFactory: ScanRepositoryItemFactory,
                 scanAPI: ScanAPI, fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._repository = repository
        self._itemFactory = itemFactory
        self._scanAPI = scanAPI
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(
            cls, repository: ScanRepository, itemFactory: ScanRepositoryItemFactory,
            scanAPI: ScanAPI,
            fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanRepositoryPresenter:
        presenter = cls(repository, itemFactory, scanAPI, fileWriterChooser)
        repository.addObserver(presenter)
        return presenter

    def __iter__(self) -> Iterator[ScanRepositoryItemPresenter]:
        for name, item in self._repository.items():
            yield ScanRepositoryItemPresenter(name, item)

    def __getitem__(self, index: int) -> ScanRepositoryItemPresenter:
        nameItemTuple = self._repository.getNameItemTupleByIndex(index)
        return ScanRepositoryItemPresenter(*nameItemTuple)

    def __len__(self) -> int:
        return len(self._repository)

    def getInitializerDisplayNameList(self) -> Sequence[str]:
        return self._itemFactory.getInitializerDisplayNameList()

    def initializeScan(self, displayName: str) -> Optional[str]:
        return self._scanAPI.insertItemIntoRepositoryFromInitializerDisplayName(displayName)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._itemFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._itemFactory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._scanAPI.insertItemIntoRepositoryFromFile(filePath, displayFileType=fileFilter)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, name: str, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, item)

        if item.getInitializer() is None:
            initializer = self._itemFactory.createFileInitializer(filePath,
                                                                  simpleFileType=fileType)

            if initializer is not None:
                item.setInitializer(initializer)

    def removeScan(self, name: str) -> None:
        self._repository.removeItem(name)

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()


class ScanPresenter(Observable, Observer):

    def __init__(self, scan: SelectedScan, scanAPI: ScanAPI,
                 dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._scan = scan
        self._scanAPI = scanAPI
        self._dataset = dataset

    @classmethod
    def createInstance(cls, scan: SelectedScan, scanAPI: ScanAPI,
                       dataset: ActiveDiffractionDataset) -> ScanPresenter:
        presenter = cls(scan, scanAPI, dataset)
        scan.addObserver(presenter)
        dataset.addObserver(presenter)
        return presenter

    def isSelectedScanValid(self) -> bool:
        selectedScan = self._scan.getSelectedItem()

        if selectedScan is None:
            return False

        datasetIndexes = set(self._dataset.getAssembledIndexes())
        scanIndexes = set(selectedScan.keys())
        return (not scanIndexes.isdisjoint(datasetIndexes))

    def selectScan(self, name: str) -> None:
        self._scan.selectItem(name)

    def getSelectedScan(self) -> str:
        return self._scan.getSelectedName()

    def getSelectableNames(self) -> Sequence[str]:
        return self._scan.getSelectableNames()

    def update(self, observable: Observable) -> None:
        if observable is self._scan:
            self.notifyObservers()
        elif observable is self._dataset:
            self.notifyObservers()


class ScanCore(StatefulCore[ScanStateData]):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 dataset: ActiveDiffractionDataset,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._builder = StreamingScanBuilder()
        self._repository = ScanRepository()
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._itemFactory = ScanRepositoryItemFactory(rng, self._settings, fileReaderChooser)
        self._itemSettingsDelegate = ScanRepositoryItemSettingsDelegate(
            self._settings, self._itemFactory, self._repository)
        self._scan = SelectedScan.createInstance(self._repository, self._itemSettingsDelegate,
                                                 settingsRegistry)
        self.sizer = ScanSizer.createInstance(self._settings, self._scan)
        self.scanAPI = ScanAPI(self._builder, self._itemFactory, self._repository, self._scan,
                               self.sizer)
        self.repositoryPresenter = ScanRepositoryPresenter.createInstance(
            self._repository, self._itemFactory, self.scanAPI, fileWriterChooser)
        self.presenter = ScanPresenter.createInstance(self._scan, self.scanAPI, dataset)

    def getStateData(self) -> ScanStateData:
        indexes: list[int] = list()
        positionXInMeters: list[float] = list()
        positionYInMeters: list[float] = list()
        selectedScan = self._scan.getSelectedItem()

        if selectedScan is not None:
            for index, point in selectedScan.untransformed.items():
                indexes.append(index)
                positionXInMeters.append(float(point.x))
                positionYInMeters.append(float(point.y))

        return ScanStateData(
            indexes=numpy.array(indexes),
            positionXInMeters=numpy.array(positionXInMeters),
            positionYInMeters=numpy.array(positionYInMeters),
        )

    def setStateData(self, stateData: ScanStateData, stateFilePath: Path) -> None:
        pointMap: dict[int, ScanPoint] = dict()

        for index, x, y in zip(stateData.indexes, stateData.positionXInMeters,
                               stateData.positionYInMeters):
            pointMap[index] = ScanPoint(x, y)

        self.scanAPI.insertItemIntoRepositoryFromScan(name='Restart',
                                                      scan=TabularScan(pointMap),
                                                      filePath=stateFilePath,
                                                      simpleFileType=stateFilePath.suffix,
                                                      selectItem=True)
