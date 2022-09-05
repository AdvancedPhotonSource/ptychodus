from __future__ import annotations
from collections.abc import ItemsView
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import (ScanFileReader, ScanFileWriter, ScanPoint, ScanPointSequence,
                         SimpleScanDictionary)
from ...api.settings import SettingsRegistry
from .initializer import ScanInitializer
from .initializerFactory import ScanInitializerFactory
from .repository import ScanRepository
from .scan import Scan
from .settings import ScanSettings
from .tabular import ScanFileInfo, TabularScanInitializer

logger = logging.getLogger(__name__)


class ScanPresenter(Observable, Observer):

    def __init__(self, initializerFactory: ScanInitializerFactory, repository: ScanRepository,
                 scan: Scan, fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        super().__init__()
        self._initializerFactory = initializerFactory
        self._repository = repository
        self._scan = scan
        self._fileWriterChooser = fileWriterChooser

    @classmethod
    def createInstance(cls, initializerFactory: ScanInitializerFactory, repository: ScanRepository,
                       scan: Scan,
                       fileWriterChooser: PluginChooser[ScanFileWriter]) -> ScanPresenter:
        presenter = cls(initializerFactory, repository, scan, fileWriterChooser)
        repository.addObserver(presenter)
        scan.addObserver(presenter)
        return presenter

    def getScanRepositoryContents(self) -> ItemsView[str, ScanInitializer]:
        return self._repository.items()

    def getActiveScanPointList(self) -> list[ScanPoint]:
        return [point for point in self._scan]

    def getActiveScan(self) -> str:
        return self._scan.name

    def canActivateScan(self, name: str) -> bool:
        canActivate = False
        initializer = self._repository.get(name)

        if isinstance(initializer, TabularScanInitializer):
            canActivate = (initializer.getFileInfo() is not None)

        return canActivate

    def setActiveScan(self, name: str) -> None:
        self._scan.setActive(name)

    def getInitializerNameList(self) -> list[str]:
        return self._initializerFactory.getInitializerNameList()

    def getInitializer(self, name: str) -> ScanInitializer:
        return self._repository[name]

    def initializeScan(self, name: str) -> None:
        initializer = self._initializerFactory.createInitializer(name)

        if initializer is None:
            logger.error(f'Unknown scan initializer \"{name}\"!')
        else:
            self._repository.insertScan(initializer)

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializerFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializerFactory.getOpenFileFilter()

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        initializerList = self._initializerFactory.openScan(filePath, fileFilter)

        for initializer in initializerList:
            self._repository.insertScan(initializer)

    def canRemoveScan(self, name: str) -> bool:
        return self._repository.canRemoveScan(name) and name != self._scan.name

    def removeScan(self, name: str) -> None:
        if self.canRemoveScan(name):
            self._repository.removeScan(name)

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

        scanDict = SimpleScanDictionary.createFromUnnamedSequence(initializer)

        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, scanDict)

        if isinstance(initializer, TabularScanInitializer):
            if initializer.getFileInfo() is None:
                fileInfo = ScanFileInfo(fileType, filePath,
                                        SimpleScanDictionary.DEFAULT_SEQUENCE_NAME)
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
        self._repository = ScanRepository()
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._initializerFactory = ScanInitializerFactory(rng, self._settings, fileReaderChooser)
        self.scan = Scan.createInstance(self._settings, self._initializerFactory, self._repository,
                                        settingsRegistry)
        self.presenter = ScanPresenter.createInstance(self._initializerFactory, self._repository,
                                                      self.scan, fileWriterChooser)

    def openScan(self, filePath: Path, fileFilter: str) -> None:
        self.presenter.openScan(filePath, fileFilter)
