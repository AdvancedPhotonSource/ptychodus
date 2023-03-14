from __future__ import annotations
from collections.abc import ItemsView
from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanArrayType, ScanFileReader, ScanFileWriter, ScanPoint, TabularScan
from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataType, StatefulCore
from .active import ActiveScan
from .api import ScanAPI
from .indexFilters import ScanIndexFilterFactory
from .itemFactory import ScanRepositoryItemFactory
from .itemRepository import ScanRepository, ScanRepositoryItem, ScanRepositoryPresenter
from .settings import ScanSettings
from .streaming import StreamingScanBuilder
from .tabular import ScanFileInfo, TabularScanRepositoryItem

logger = logging.getLogger(__name__)


class ScanPresenter(Observable, Observer):

    def __init__(self, itemFactory: ScanRepositoryItemFactory, repository: ScanRepository,
                 scan: ActiveScan, scanAPI: ScanAPI,
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._itemFactory = itemFactory
        self._repository = repository
        self._scan = scan
        self._scanAPI = scanAPI
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, itemFactory: ScanRepositoryItemFactory, repository: ScanRepository,
                       scan: ActiveScan, scanAPI: ScanAPI,
                       fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanPresenter:
        presenter = cls(itemFactory, repository, scan, scanAPI, fileWriterChooser)
        scan.addObserver(presenter)
        return presenter

    def getScanRepositoryContents(self) -> ItemsView[str, ScanRepositoryItem]:  # FIXME remove
        return self._repository.items()

    def isActiveScanValid(self) -> bool:
        # FIXME isActiveScanValid should require more than one common index with active dataset
        return (len(self._scan) > 1)

    def getActiveScanPointList(self) -> list[ScanPoint]:
        return [point for point in self._scan.values()]

    def getActiveScan(self) -> str:
        return self._scan.name

    def canActivateScan(self, name: str) -> bool:
        return self._scan.canActivateScan(name)

    def setActiveScan(self, name: str) -> None:
        self._scanAPI.setActiveScan(name)

    def getItemNameList(self) -> list[str]:
        return self._itemFactory.getItemNameList()

    def getItem(self, name: str) -> ScanRepositoryItem:  # FIXME remove
        return self._repository[name]

    def initializeScan(self, name: str) -> Optional[str]:
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
            initializer = self._repository[name]
        except KeyError:
            logger.error(f'Unable to locate \"{name}\"!')
            return

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, [initializer])

        if isinstance(initializer, TabularScanRepositoryItem):
            if initializer.getFileInfo() is None:
                fileInfo = ScanFileInfo(fileType, filePath)
                initializer.setFileInfo(fileInfo)

    def update(self, observable: Observable) -> None:
        if observable is self._scan:
            self.notifyObservers()


class ScanCore(StatefulCore):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
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
        self.scanAPI = ScanAPI(self._builder, self._itemFactory, self._repository, self.scan)
        self.repositoryPresenter = ScanRepositoryPresenter.createInstance(self._repository)
        self.presenter = ScanPresenter.createInstance(self._itemFactory, self._repository,
                                                      self.scan, self.scanAPI, fileWriterChooser)

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
