from __future__ import annotations
from collections.abc import ItemsView, Iterator
from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanArrayType, ScanFileReader, ScanFileWriter, ScanPoint, TabularScan
from ...api.settings import SettingsRegistry
from ..data import ActiveDiffractionDataset
from ..statefulCore import StateDataType, StatefulCore
from .active import ActiveScan
from .api import ScanAPI
from .indexFilters import ScanIndexFilterFactory
from .itemFactory import ScanRepositoryItemFactory
from .itemRepository import ScanRepository, TransformedScanRepositoryItem
from .settings import ScanSettings
from .sizer import ScanSizer
from .streaming import StreamingScanBuilder
from .tabular import ScanFileInfo, TabularScanRepositoryItem

logger = logging.getLogger(__name__)


class ScanRepositoryPresenter(Observable, Observer):

    def __init__(self, repository: ScanRepository, itemFactory: ScanRepositoryItemFactory,
                 scanAPI: ScanAPI, fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._repository = repository
        self._itemFactory = itemFactory
        self._scanAPI = scanAPI
        self._fileWriterChooser = fileWriterChooser
        self._itemNameList: list[str] = list()

    @classmethod
    def createInstance(
            cls, repository: ScanRepository, itemFactory: ScanRepositoryItemFactory,
            scanAPI: ScanAPI,
            fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanRepositoryPresenter:
        presenter = cls(repository, itemFactory, scanAPI, fileWriterChooser)
        presenter._updateItemPresenterList()
        repository.addObserver(presenter)
        return presenter

    def __iter__(self) -> Iterator[TransformedScanRepositoryItem]:
        return iter(self._repository.values())

    def __getitem__(self, index: int) -> TransformedScanRepositoryItem:
        itemName = self._itemNameList[index]
        return self._repository[itemName]

    def __len__(self) -> int:
        return len(self._itemNameList)

    def getInitializerNameList(self) -> list[str]:
        return self._itemFactory.getInitializerNameList()

    def initializeScan(self, name: str) -> list[str]:
        return self._scanAPI.insertScanIntoRepositoryFromInitializer(name)

    def getOpenFileFilterList(self) -> list[str]:
        return self._itemFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._itemFactory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self._scanAPI.insertScanIntoRepositoryFromFile(filePath, fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
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
        writer.write(filePath, [item])

        innerItem = item._item

        # TODO test this
        if isinstance(innerItem, TabularScanRepositoryItem):
            if innerItem.getFileInfo() is None:
                fileInfo = ScanFileInfo(fileType, filePath)
                innerItem.setFileInfo(fileInfo)

    def canRemoveScan(self, name: str) -> bool:
        return self._repository.canRemoveItem(name)

    def removeScan(self, name: str) -> None:
        self._repository.removeItem(name)

    def _updateItemPresenterList(self) -> None:
        itemNameList: list[str] = list()

        for itemName, scan in self._repository.items():
            scan.name = itemName
            itemNameList.append(itemName)

        self._itemNameList = itemNameList
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self._updateItemPresenterList()


class ScanPresenter(Observable, Observer):

    def __init__(self, scan: ActiveScan, scanAPI: ScanAPI,
                 dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._scan = scan
        self._scanAPI = scanAPI
        self._dataset = dataset

    @classmethod
    def createInstance(cls, scan: ActiveScan, scanAPI: ScanAPI,
                       dataset: ActiveDiffractionDataset) -> ScanPresenter:
        presenter = cls(scan, scanAPI, dataset)
        scan.addObserver(presenter)
        dataset.addObserver(presenter)
        return presenter

    def isActiveScanValid(self) -> bool:
        datasetIndexes = set(self._dataset.getAssembledIndexes())
        scanIndexes = set(self._scan.keys())
        return (not scanIndexes.isdisjoint(datasetIndexes))

    def setActiveScan(self, name: str) -> None:
        self._scanAPI.setActiveScan(name)

    def getActiveScan(self) -> str:
        return self._scan.name

    def update(self, observable: Observable) -> None:
        if observable is self._scan:
            self.notifyObservers()
        elif observable is self._dataset:
            self.notifyObservers()


class ScanCore(StatefulCore):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 dataset: ActiveDiffractionDataset,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._builder = StreamingScanBuilder()
        self._repository = ScanRepository()
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._indexFilterFactory = ScanIndexFilterFactory()
        self._itemFactory = ScanRepositoryItemFactory(rng, self._settings,
                                                      self._indexFilterFactory, fileReaderChooser)
        self.scan = ActiveScan.createInstance(self._settings, self._itemFactory, self._repository,
                                              settingsRegistry)
        self.sizer = ScanSizer.createInstance(self._settings, self.scan)
        self.scanAPI = ScanAPI(self._builder, self._itemFactory, self._repository, self.scan)
        self.repositoryPresenter = ScanRepositoryPresenter.createInstance(
            self._repository, self._itemFactory, self.scanAPI, fileWriterChooser)
        self.presenter = ScanPresenter.createInstance(self.scan, self.scanAPI, dataset)

    def getStateData(self, *, restartable: bool) -> StateDataType:
        scanIndex: list[int] = list()
        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for index, point in self.scan.untransformed.items():
            scanIndex.append(index)
            scanXInMeters.append(float(point.x))
            scanYInMeters.append(float(point.y))

        state: StateDataType = {
            'scanIndex': numpy.array(scanIndex),
            'scanXInMeters': numpy.array(scanXInMeters),
            'scanYInMeters': numpy.array(scanYInMeters),
        }

        return state

    def setStateData(self, state: StateDataType) -> None:
        pointMap: dict[int, ScanPoint] = dict()
        scanIndex = state['scanIndex']
        scanXInMeters = state['scanXInMeters']
        scanYInMeters = state['scanYInMeters']

        for index, x, y in zip(scanIndex, scanXInMeters, scanYInMeters):
            pointMap[index] = ScanPoint(
                x=Decimal(repr(x)),
                y=Decimal(repr(y)),
            )

        scan = TabularScan('Restart', pointMap)
        itemName = self.scanAPI.insertScanIntoRepository(scan, ScanFileInfo.createNull())
        self.scanAPI.setActiveScan(itemName)
