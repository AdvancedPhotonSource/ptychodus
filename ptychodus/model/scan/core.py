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
from .active import ActiveScan
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository
from .repositoryItem import ScanRepositoryItem
from .settings import ScanSettings
from .tabular import ScanFileInfo, TabularScanRepositoryItem

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanRepositoryKeyAndValue:
    name: str
    item: ScanRepositoryItem


class ScanPresenter(Observable, Observer):

    def __init__(self, initializerFactory: ScanRepositoryItemFactory, repository: ScanRepository,
                 scan: ActiveScan, fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._scan = scan
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, initializerFactory: ScanRepositoryItemFactory,
                       repository: ScanRepository, scan: ActiveScan,
                       fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanPresenter:
        presenter = cls(initializerFactory, repository, scan, fileWriterChooser)
        repository.addObserver(presenter)
        scan.addObserver(presenter)
        return presenter

    def getScanRepositoryContents(self) -> ItemsView[str, Scan]:
        return self._repository.items()

    def getScanRepositoryKeysAndValues(self) -> list[ScanRepositoryKeyAndValue]:
        return [ScanRepositoryKeyAndValue(name, item) for name, item in self._repository.items()]

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
        self._scan.setActiveScan(name)

    def getInitializerNameList(self) -> list[str]:
        return self._initializerFactory.getInitializerNameList()

    def getInitializer(self, name: str) -> ScanRepositoryItem:
        return self._repository[name]

    def initializeScan(self, name: str) -> None:
        initializer = self._initializerFactory.createInitializer(name)

        if initializer is None:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            self._repository.insertItem(initializer)

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializerFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializerFactory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        initializerList = self._initializerFactory.openScan(filePath, fileFilter)

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

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()


class ScanCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self.repository = ScanRepository()
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self.initializerFactory = ScanRepositoryItemFactory(rng, self._settings, fileReaderChooser)
        self.scan = ActiveScan.createInstance(self._settings, self.initializerFactory,
                                              self.repository, settingsRegistry)
        self.presenter = ScanPresenter.createInstance(self.initializerFactory, self.repository,
                                                      self.scan, fileWriterChooser)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.presenter.openScan(filePath, fileFilter)

    def getScanArrayInMeters(self) -> ScanArrayType:
        # TODO use indexes
        scanXInMeters = [float(point.x) for point in self.scan.values()]
        scanYInMeters = [float(point.y) for point in self.scan.values()]
        return numpy.column_stack((scanYInMeters, scanXInMeters))

    def setScanArrayInMeters(self, array: ScanArrayType) -> None:
        pointList: list[ScanPoint] = list()

        # TODO use indexes
        for row in array:
            point = ScanPoint(
                x=Decimal(repr(row[1])),
                y=Decimal(repr(row[0])),
            )
            pointList.append(point)

        name = 'Restart'
        scan = TabularScan.createFromPointSequence(name, pointList)
        item = self.initializerFactory.createTabularInitializer(scan, None)
        self.repository.insertItem(item)
        self.scan.setActiveScan(name)
