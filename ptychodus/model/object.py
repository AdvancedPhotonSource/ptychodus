from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
import logging
import threading

import numpy

from ..api.geometry import Box, Interval
from ..api.object import *
from ..api.observer import Observable, Observer
from ..api.plugins import PluginChooser, PluginEntry
from ..api.settings import SettingsRegistry, SettingsGroup
from .data import CropSizer, Detector
from .image import ImageExtent
from .probe import ProbeSizer
from .scan import Scan

logger = logging.getLogger(__name__)


class ObjectSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Random')
        self.inputFileType = settingsGroup.createStringEntry('InputFileType', 'NPY')
        self.inputFilePath = settingsGroup.createPathEntry('InputFilePath',
                                                           Path('/path/to/object.npy'))

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ObjectSettings:
        settings = cls(settingsRegistry.createGroup('Object'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ObjectSizer(Observable, Observer):

    def __init__(self, detector: Detector, cropSizer: CropSizer, scan: Scan,
                 probeSizer: ProbeSizer) -> None:
        super().__init__()
        self._detector = detector
        self._cropSizer = cropSizer
        self._scan = scan
        self._probeSizer = probeSizer

    @classmethod
    def createInstance(cls, detector: Detector, cropSizer: CropSizer, scan: Scan,
                       probeSizer: ProbeSizer) -> ObjectSizer:
        sizer = cls(detector, cropSizer, scan, probeSizer)
        detector.addObserver(sizer)
        cropSizer.addObserver(sizer)
        scan.addObserver(sizer)
        probeSizer.addObserver(sizer)
        return sizer

    @property
    def _lambdaZ_m2(self) -> Decimal:
        return self._probeSizer.getWavelengthInMeters() \
                * self._detector.getDetectorDistanceInMeters()

    def getPixelSizeXInMeters(self) -> Decimal:
        extentXInMeters = self._cropSizer.getExtentXInPixels() \
                * self._detector.getPixelSizeXInMeters()
        return self._lambdaZ_m2 / extentXInMeters

    def getPixelSizeYInMeters(self) -> Decimal:
        extentYInMeters = self._cropSizer.getExtentYInPixels() \
                * self._detector.getPixelSizeYInMeters()
        return self._lambdaZ_m2 / extentYInMeters

    def getScanExtent(self) -> ImageExtent:
        scanExtent = ImageExtent(0, 0)
        xint_m = None
        yint_m = None

        for point in self._scan:
            if xint_m and yint_m:
                xint_m.hull(point.x)
                yint_m.hull(point.y)
            else:
                xint_m = Interval[Decimal](point.x, point.x)
                yint_m = Interval[Decimal](point.y, point.y)

        if xint_m and yint_m:
            xint_px = xint_m.length / self.getPixelSizeXInMeters()
            yint_px = yint_m.length / self.getPixelSizeYInMeters()

            scanExtent = ImageExtent(width=int(xint_px.to_integral_exact(rounding=ROUND_CEILING)),
                                     height=int(yint_px.to_integral_exact(rounding=ROUND_CEILING)))

        return scanExtent

    def getPaddingExtent(self) -> ImageExtent:
        return 2 * (self._probeSizer.getProbeExtent() // 2)

    def getObjectExtent(self) -> ImageExtent:
        return self.getScanExtent() + self.getPaddingExtent()

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
        elif observable is self._cropSizer:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()


class UniformRandomObjectInitializer:

    def __init__(self, rng: numpy.random.Generator, sizer: ObjectSizer) -> None:
        self._rng = rng
        self._sizer = sizer

    def __call__(self) -> ObjectArrayType:
        size = self._sizer.getObjectExtent().shape
        magnitude = numpy.sqrt(self._rng.uniform(low=0., high=1., size=size))
        phase = self._rng.uniform(low=0., high=2. * numpy.pi, size=size)
        return magnitude * numpy.exp(1.j * phase)


class FileObjectInitializer(Observer):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer,
                 fileReaderChooser: PluginChooser[ObjectFileReader]) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._fileReaderChooser = fileReaderChooser
        self._array = numpy.zeros(sizer.getObjectExtent().shape, dtype=complex)

    @classmethod
    def createInstance(
            cls, settings: ObjectSettings, sizer: ObjectSizer,
            fileReaderChooser: PluginChooser[ObjectFileReader]) -> FileObjectInitializer:
        initializer = cls(settings, sizer, fileReaderChooser)

        settings.inputFileType.addObserver(initializer)
        initializer._fileReaderChooser.addObserver(initializer)
        initializer._syncFileReaderFromSettings()

        settings.inputFilePath.addObserver(initializer)
        initializer._openObjectFromSettings()

        return initializer

    def __call__(self) -> ObjectArrayType:
        return self._array

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.getCurrentDisplayName()

    def _syncFileReaderFromSettings(self) -> None:
        self._fileReaderChooser.setFromSimpleName(self._settings.inputFileType.value)

    def _syncFileReaderToSettings(self) -> None:
        self._settings.inputFileType.value = self._fileReaderChooser.getCurrentSimpleName()

    def _openObject(self, filePath: Path) -> None:
        if filePath is not None and filePath.is_file():
            logger.debug(f'Reading {filePath}')
            fileReader = self._fileReaderChooser.getCurrentStrategy()
            self._array = fileReader.read(filePath)

    def openObject(self, filePath: Path, fileFilter: str) -> None:
        self._fileReaderChooser.setFromDisplayName(fileFilter)

        if self._settings.inputFilePath.value == filePath:
            self._openObject(filePath)

        self._settings.inputFilePath.value = filePath

    def _openObjectFromSettings(self) -> None:
        self._openObject(self._settings.inputFilePath.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.inputFileType:
            self._syncFileReaderFromSettings()
        elif observable is self._fileReaderChooser:
            self._syncFileReaderToSettings()
        elif observable is self._settings.inputFilePath:
            self._openObjectFromSettings()


class Object(Observable):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._array = numpy.zeros(sizer.getObjectExtent().shape, dtype=complex)
        self._arrayLock = threading.Lock()

    def getArray(self) -> ObjectArrayType:
        return self._array

    def setArray(self, array: ObjectArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Object must be a complex-valued ndarray')

        with self._arrayLock:
            self._array = array

        self.notifyObservers()


class ObjectInitializer(Observable, Observer):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer, object_: Object,
                 fileInitializer: FileObjectInitializer,
                 fileWriterChooser: PluginChooser[ObjectFileWriter],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._object = object_
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable
        self._fileInitializer = fileInitializer
        self._initializerChooser = PluginChooser[ObjectInitializerType](
            PluginEntry[ObjectInitializerType](simpleName='FromFile',
                                               displayName='From File',
                                               strategy=self._fileInitializer))

    @classmethod
    def createInstance(cls, rng: numpy.random.Generator, settings: ObjectSettings,
                       sizer: ObjectSizer, object_: Object, fileInitializer: FileObjectInitializer,
                       fileWriterChooser: PluginChooser[ObjectFileWriter],
                       reinitObservable: Observable) -> ObjectInitializer:
        initializer = cls(settings, sizer, object_, fileInitializer, fileWriterChooser,
                          reinitObservable)

        urandInit = PluginEntry[ObjectInitializerType](simpleName='Random',
                                                       displayName='Random',
                                                       strategy=UniformRandomObjectInitializer(
                                                           rng, sizer))
        initializer._initializerChooser.addStrategy(urandInit)

        settings.initializer.addObserver(initializer)
        initializer._initializerChooser.addObserver(initializer)
        initializer._syncInitializerFromSettings()
        reinitObservable.addObserver(initializer)

        return initializer

    def getInitializerList(self) -> list[str]:
        return self._initializerChooser.getDisplayNameList()

    def getInitializer(self) -> str:
        return self._initializerChooser.getCurrentDisplayName()

    def setInitializer(self, name: str) -> None:
        self._initializerChooser.setFromDisplayName(name)

    def initializeObject(self) -> None:
        initializer = self._initializerChooser.getCurrentStrategy()
        simpleName = self._initializerChooser.getCurrentSimpleName()
        logger.debug(f'Initializing {simpleName} Object')
        self._object.setArray(initializer())

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileInitializer.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._fileInitializer.getOpenFileFilter()

    def openObject(self, filePath: Path, fileFilter: str) -> None:
        self._fileInitializer.openObject(filePath, fileFilter)
        self._initializerChooser.setToDefault()
        self.initializeObject()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.getCurrentDisplayName()

    def saveObject(self, filePath: Path, fileFilter: str) -> None:
        logger.debug(f'Writing {filePath}')
        self._fileWriterChooser.setFromDisplayName(fileFilter)
        writer = self._fileWriterChooser.getCurrentStrategy()
        writer.write(filePath, self._object.getArray())

    def _syncInitializerFromSettings(self) -> None:
        self._initializerChooser.setFromSimpleName(self._settings.initializer.value)

    def _syncInitializerToSettings(self) -> None:
        self._settings.initializer.value = self._initializerChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self._syncInitializerFromSettings()
        elif observable is self._initializerChooser:
            self._syncInitializerToSettings()
        elif observable is self._reinitObservable:
            self.initializeObject()


class ObjectPresenter(Observable, Observer):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer, obj: Object,
                 initializer: ObjectInitializer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._object = obj
        self._initializer = initializer

    @classmethod
    def createInstance(cls, settings: ObjectSettings, sizer: ObjectSizer, obj: Object,
                       initializer: ObjectInitializer) -> ObjectPresenter:
        presenter = cls(settings, sizer, obj, initializer)
        settings.addObserver(presenter)
        sizer.addObserver(presenter)
        obj.addObserver(presenter)
        initializer.addObserver(presenter)
        return presenter

    def getInitializerList(self) -> list[str]:
        return self._initializer.getInitializerList()

    def getInitializer(self) -> str:
        return self._initializer.getInitializer()

    def setInitializer(self, name: str) -> None:
        self._initializer.setInitializer(name)

    def getOpenFileFilterList(self) -> list[str]:
        return self._initializer.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._initializer.getOpenFileFilter()

    def openObject(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.openObject(filePath, fileFilter)

    def getSaveFileFilterList(self) -> list[str]:
        return self._initializer.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._initializer.getSaveFileFilter()

    def saveObject(self, filePath: Path, fileFilter: str) -> None:
        self._initializer.saveObject(filePath, fileFilter)

    def initializeObject(self) -> None:
        self._initializer.initializeObject()

    def getObject(self) -> ObjectArrayType:
        return self._object.getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._sizer:
            self.notifyObservers()
        elif observable is self._object:
            self.notifyObservers()
        elif observable is self._initializer:
            self.notifyObservers()
