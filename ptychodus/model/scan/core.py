from __future__ import annotations
from collections.abc import ItemsView
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import Scan, ScanArrayType, ScanFileReader, ScanFileWriter, ScanPoint, TabularScan
from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataType, StatefulCore
from .active import ActiveScan
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository
from .repositoryItem import ScanRepositoryItem
from .settings import ScanSettings
from .streaming import StreamingScanBuilder
from .tabular import ScanFileInfo, TabularScanRepositoryItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanRepositoryKeyAndValue:
    name: str
    item: ScanRepositoryItem


class ScanPresenter(Observable, Observer):

    def __init__(self, factory: ScanRepositoryItemFactory, repository: ScanRepository,
                 scan: ActiveScan, builder: StreamingScanBuilder,
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._factory = factory
        self._repository = repository
        self._scan = scan
        self._builder = builder
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, factory: ScanRepositoryItemFactory, repository: ScanRepository,
                       scan: ActiveScan, builder: StreamingScanBuilder,
                       fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanPresenter:
        presenter = cls(factory, repository, scan, builder, fileWriterChooser)
        repository.addObserver(presenter)
        scan.addObserver(presenter)
        return presenter

    def getScanRepositoryContents(self) -> ItemsView[str, Scan]:
        return self._repository.items()

    def getScanRepositoryKeysAndValues(self) -> list[ScanRepositoryKeyAndValue]:
        return [ScanRepositoryKeyAndValue(name, item) for name, item in self._repository.items()]

    def isActiveScanValid(self) -> bool:
        # TODO isActiveScanValid should require more than one common index with active dataset
        return (len(self._scan) > 1)

    def getActiveScanPointList(self) -> list[ScanPoint]:
        return [point for point in self._scan.values()]

    def getActiveScan(self) -> str:
        return self._scan.name

    def canActivateScan(self, name: str) -> bool:
        return self._scan.canActivateScan(name)

    def setActiveScan(self, name: str) -> None:
        self._scan.setActiveScan(name)

    def getItemNameList(self) -> list[str]:
        return self._factory.getItemNameList()

    def getItem(self, name: str) -> ScanRepositoryItem:
        return self._repository[name]

    def initializeScan(self, name: str) -> None:
        scan = self._factory.createItem(name)

        if scan is None:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            self._repository.insertItem(scan)

    def getOpenFileFilterList(self) -> list[str]:
        return self._factory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._factory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        initializerList = self._factory.openScan(filePath, fileFilter)

        for initializer in initializerList:
            self._repository.insertItem(initializer)

    def canRemoveScan(self, name: str) -> bool:
        return self._repository.canRemoveItem(name) and name != self._scan.name

    def removeScan(self, name: str) -> None:
        if self.canRemoveScan(name):
            self._repository.removeItem(name)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveScan(self, filePath: Path, fileFilter: str, name: str) -> None:
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

    def initializeStreamingScan(self) -> None:
        self._builder.reset()

    def insertArrayTimeStamp(self, arrayIndex: int, timeStamp: float) -> None:
        self._builder.insertArrayTimeStamp(arrayIndex, timeStamp)

    def assembleScanPositionsX(self, valuesInMeters: list[float], timeStamps: list[float]) -> None:
        self._builder.assembleScanPositionsX(valuesInMeters, timeStamps)

    def assembleScanPositionsY(self, valuesInMeters: list[float], timeStamps: list[float]) -> None:
        self._builder.assembleScanPositionsY(valuesInMeters, timeStamps)

    def finalizeStreamingScan(self) -> None:
        scan = self._builder.build()
        name = self._repository.insertItem(scan)
        self._scan.setActiveScan(name)

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()


class ScanCore(StatefulCore):

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self.repository = ScanRepository()
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self.factory = ScanRepositoryItemFactory(rng, self._settings, fileReaderChooser)
        self._builder = StreamingScanBuilder(self.factory)
        self.scan = ActiveScan.createInstance(self._settings, self.factory, self.repository,
                                              settingsRegistry)
        self.presenter = ScanPresenter.createInstance(self.factory, self.repository, self.scan,
                                                      self._builder, fileWriterChooser)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.presenter.openScan(filePath, fileFilter)

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

        name = 'Restart'
        scan = TabularScan(name, pointMap)
        item = self.factory.createTabularItem(scan, None)
        self.repository.insertItem(item)
        self.scan.setActiveScan(name)
