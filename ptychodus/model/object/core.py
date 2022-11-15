from __future__ import annotations
from decimal import Decimal
from pathlib import Path
import logging

import numpy

from ...api.object import ObjectArrayType, ObjectFileReader, ObjectFileWriter
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry
from ...api.settings import SettingsRegistry
from ..data import CropSizer
from ..detector import Detector
from ..probe import Apparatus, ProbeSizer
from ..scan import Scan
from .file import FileObjectInitializer
from .initializer import ObjectInitializer
from .object import Object
from .settings import ObjectSettings
from .sizer import ObjectSizer
from .uniform import UniformObjectInitializer
from .urand import UniformRandomObjectInitializer

logger = logging.getLogger(__name__)


class ObjectPresenter(Observable, Observer):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer, apparatus: Apparatus,
                 object_: Object, initializerChooser: PluginChooser[ObjectInitializer],
                 fileWriterChooser: PluginChooser[ObjectFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._apparatus = apparatus
        self._object = object_
        self._initializerChooser = initializerChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: ObjectSettings, sizer: ObjectSizer, apparatus: Apparatus,
                       object_: Object, initializerChooser: PluginChooser[ObjectInitializer],
                       fileWriterChooser: PluginChooser[ObjectFileWriter],
                       reinitObservable: Observable) -> ObjectPresenter:
        presenter = cls(settings, sizer, apparatus, object_, initializerChooser, fileWriterChooser,
                        reinitObservable)

        settings.addObserver(presenter)
        sizer.addObserver(presenter)
        apparatus.addObserver(presenter)
        object_.addObserver(presenter)
        reinitObservable.addObserver(presenter)

        presenter._syncFromSettings()

        return presenter

    def isActiveObjectValid(self) -> bool:
        actualExtent = self._object.getObjectExtent()
        expectedExtent = self._sizer.getObjectExtent()
        widthIsBigEnough = (actualExtent.width >= expectedExtent.width)
        heightIsBigEnough = (actualExtent.height >= expectedExtent.height)
        return (widthIsBigEnough and heightIsBigEnough)

    def initializeObject(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Object')
        initializer.syncToSettings(self._settings)
        self._object.setArray(initializer())

    def getInitializerNameList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializerName(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializerByName(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def getInitializer(self) -> ObjectInitializer:
        return self._initializerChooser.getCurrentStrategy()

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
        return self._apparatus.getObjectPlanePixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeYInMeters()

    def getObject(self) -> ObjectArrayType:
        return self._object.getArray()

    def _syncFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)
        initializer = self._initializerChooser.getCurrentStrategy()
        initializer.syncFromSettings(self._settings)
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncFromSettings()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._apparatus:
            self.notifyObservers()
        elif observable is self._object:
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self.initializeObject()


class ObjectCore:

    @staticmethod
    def _createInitializerChooser(
            rng: numpy.random.Generator, settings: ObjectSettings, sizer: ObjectSizer,
            fileReaderChooser: PluginChooser[ObjectFileReader]
    ) -> PluginChooser[ObjectInitializer]:
        initializerList = [
            FileObjectInitializer.createInstance(settings, sizer, fileReaderChooser),
            UniformObjectInitializer.createInstance(settings, sizer),
            UniformRandomObjectInitializer(rng, sizer),
        ]

        pluginList = [
            PluginEntry[ObjectInitializer](simpleName=ini.simpleName,
                                           displayName=ini.displayName,
                                           strategy=ini) for ini in initializerList
        ]

        return PluginChooser[ObjectInitializer].createFromList(pluginList)

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 apparatus: Apparatus, scan: Scan, probeSizer: ProbeSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader],
                 fileWriterChooser: PluginChooser[ObjectFileWriter]) -> None:
        self.settings = ObjectSettings.createInstance(settingsRegistry)
        self.sizer = ObjectSizer.createInstance(apparatus, scan, probeSizer)
        self.object = Object(self.sizer)

        self._initializerChooser = ObjectCore._createInitializerChooser(
            rng, self.settings, self.sizer, fileReaderChooser)

        self.presenter = ObjectPresenter.createInstance(self.settings, self.sizer, apparatus,
                                                        self.object, self._initializerChooser,
                                                        fileWriterChooser, settingsRegistry)
