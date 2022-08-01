from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ...api.object import (ObjectArrayType, ObjectFileReader, ObjectFileWriter,
                           ObjectInitializerType)
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from ...api.settings import SettingsRegistry
from ..data import CropSizer, Detector
from ..probe import ProbeSizer
from ..scan import Scan
from .file import FileObjectInitializer
from .object import Object
from .settings import ObjectSettings
from .sizer import ObjectSizer
from .urand import UniformRandomObjectInitializer

logger = logging.getLogger(__name__)


class ObjectPresenter(Observable, Observer):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer, object_: Object,
                 initializerChooser: PluginChooser[ObjectInitializerType],
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._object = object_
        self._initializerChooser = initializerChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: ObjectSettings, sizer: ObjectSizer, object_: Object,
                       initializerChooser: PluginChooser[ObjectInitializerType],
                       fileReaderChooser: PluginChooser[ObjectFileReader],
                       fileWriterChooser: PluginChooser[ObjectFileWriter],
                       reinitObservable: Observable) -> ObjectPresenter:
        presenter = cls(settings, sizer, object_, initializerChooser, fileReaderChooser,
                        fileWriterChooser, reinitObservable)

        settings.addObserver(presenter)
        sizer.addObserver(presenter)
        object_.addObserver(presenter)
        initializerChooser.addObserver(presenter)
        fileReaderChooser.addObserver(presenter)
        reinitObservable.addObserver(presenter)

        presenter._syncFromSettings()

        return presenter

    def initializeObject(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Object')
        self._object.setArray(initializer())

    def getInitializerNameList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializer(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializer(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def getOpenFilePath(self) -> Path:
        return self._settings.inputFilePath.value

    def setOpenFilePath(self, filePath: Path) -> None:
        self._settings.inputFilePath.value = filePath

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def setOpenFileFilter(self, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveObject(self, filePath: Path, fileFilter: str) -> None:
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        fileType = self._fileWriterChooser.getCurrentSimpleName()
        writer = self._fileWriterChooser.getCurrentStrategy()

        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer.write(filePath, self._object.getArray())

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._sizer.getPixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._sizer.getPixelSizeYInMeters()

    def getObject(self) -> ObjectArrayType:
        return self._object.getArray()

    def _syncFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)
        self.notifyObservers()

    def _syncInitializerToSettings(self) -> None:
        self._settings.initializer.value = self._initializerChooser.getCurrentSimpleName()

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncFromSettings()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._object:
            self.notifyObservers()
        elif observable is self._initializerChooser:
            self._syncInitializerToSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._reinitObservable:
            self.initializeObject()


class ObjectCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 detector: Detector, cropSizer: CropSizer, scan: Scan, probeSizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self.settings = ObjectSettings.createInstance(settingsRegistry)
        self.sizer = ObjectSizer.createInstance(detector, cropSizer, scan, probeSizer)
        self.object = Object(self.sizer)

        self._filePlugin = PluginEntry[ObjectInitializerType](
            simpleName='FromFile',
            displayName='Open File...',
            strategy=FileObjectInitializer(self.settings, self.sizer, fileReaderChooser),
        )
        self._urandPlugin = PluginEntry[ObjectInitializerType](
            simpleName='Random',
            displayName='Random',
            strategy=UniformRandomObjectInitializer(rng, self.sizer),
        )
        self._initializerChooser = PluginChooser[ObjectInitializerType].createFromList(
            [self._filePlugin, self._urandPlugin])

        self.presenter = ObjectPresenter.createInstance(self.settings, self.sizer, self.object,
                                                        self._initializerChooser,
                                                        fileReaderChooser, fileWriterChooser,
                                                        settingsRegistry)
