from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Callable
import logging

import numpy

from .crop import CropSizer
from .detector import Detector
from .geometry import Box, Interval
from .image import ImageExtent
from .observer import Observable, Observer
from .probe import ProbeSizer
from .scan import Scan
from .settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class ObjectSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.initializer = settingsGroup.createStringEntry('Initializer', 'Random')
        self.customFilePath = settingsGroup.createPathEntry('CustomFilePath', None)

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
                * self._detector.distanceToObjectInMeters

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._lambdaZ_m2 / self._cropSizer.getExtentXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._lambdaZ_m2 / self._cropSizer.getExtentYInMeters()

    def getScanExtent(self) -> ImageExtent:
        scanBox_m = self._scan.getBoundingBoxInMeters()
        scanWidth_px = 0
        scanHeight_px = 0

        if scanBox_m:
            assert len(scanBox_m) == 2

            scanWidth_px = scanBox_m[0].length / self.getPixelSizeXInMeters()
            scanWidth_px = int(numpy.ceil(scanWidth_px))

            scanHeight_px = scanBox_m[1].length / self.getPixelSizeYInMeters()
            scanHeight_px = int(numpy.ceil(scanHeight_px))

        return ImageExtent(width=scanWidth_px, height=scanHeight_px)

    def getPaddingExtent(self) -> ImageExtent:
        pad = 2 * (self._probeSizer.getProbeSize() // 2)
        return ImageExtent(width=pad, height=pad)

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
    def __init__(self, sizer: ObjectSizer, rng: numpy.random.Generator) -> None:
        self._sizer = sizer
        self._rng = rng

    def __call__(self) -> numpy.ndarray:
        size = self._sizer.getObjectExtent().shape
        magnitude = numpy.sqrt(self._rng.uniform(low=0., high=1., size=size))
        phase = self._rng.uniform(low=0., high=2. * numpy.pi, size=size)
        return magnitude * numpy.exp(1.j * phase)

    def __str__(self) -> str:
        return 'Random'


class CustomObjectInitializer:
    def __init__(self, sizer: ObjectSizer) -> None:
        self._sizer = sizer
        self._array = numpy.zeros(sizer.getObjectExtent().shape, dtype=complex)

    def setArray(self, array: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Object must be a complex-valued ndarray')

        self._array = array

    def __call__(self) -> numpy.ndarray:
        return self._array

    def __str__(self) -> str:
        return 'Custom'


class Object(Observable):
    FILE_FILTER = 'NumPy Binary Files (*.npy)'

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._array = numpy.zeros(sizer.getObjectExtent().shape, dtype=complex)

    def getArray(self) -> numpy.ndarray:
        return self._array

    def setArray(self, array: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Object must be a complex-valued ndarray')

        self._array = array
        self.notifyObservers()

    def read(self, filePath: Path) -> None:
        self._settings.customFilePath.value = filePath
        array = numpy.load(filePath)
        self.setArray(array)

    def write(self, filePath: Path) -> None:
        numpy.save(filePath, self._array)


class ObjectInitializer(Observable, Observer):
    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer, obj: Object,
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._object = obj
        self._reinitObservable = reinitObservable
        self._customInitializer = CustomObjectInitializer(sizer)
        self._initializer = self._customInitializer
        self._initializerList: list[Callable[[], numpy.ndarray]] = [self._customInitializer]

    @classmethod
    def createInstance(cls, rng: numpy.random.Generator, detectorSettings: DetectorSettings,
                       objectSettings: ObjectSettings, sizer: ObjectSizer, obj: Object,
                       reinitObservable: Observable) -> ObjectInitializer:
        initializer = cls(objectSettings, sizer, obj, reinitObservable)

        urandInit = UniformRandomObjectInitializer(sizer, rng)
        initializer.addInitializer(urandInit)

        initializer.setInitializerFromSettings()
        objectSettings.initializer.addObserver(initializer)
        reinitObservable.addObserver(initializer)

        return initializer

    def addInitializer(self, initializer: Callable[[], numpy.ndarray]) -> None:
        self._initializerList.append(initializer)

    def getInitializerList(self) -> list[str]:
        return [str(initializer) for initializer in self._initializerList]

    def getInitializer(self) -> str:
        return str(self._initializer)

    def _setInitializer(self, initializer: Callable[[], numpy.ndarray]) -> None:
        if initializer is not self._initializer:
            self._initializer = initializer
            self._settings.initializer.value = str(self._initializer)
            self.notifyObservers()

    def setInitializer(self, name: str) -> None:
        try:
            initializer = next(ini for ini in self._initializerList
                               if name.casefold() == str(ini).casefold())
        except StopIteration:
            logger.debug(f'Invalid initializer \"{name}\"')
            return

        self._setInitializer(initializer)

    def setInitializerFromSettings(self) -> None:
        self.setInitializer(self._settings.initializer.value)

    def initializeObject(self) -> None:
        self._object.setArray(self._initializer())

    def openObject(self, filePath: Path) -> None:
        self._settings.customFilePath.value = filePath
        self._object.read(filePath)
        self._customInitializer.setArray(self._object.getArray())
        self._setInitializer(self._customInitializer)

    def _preloadObjectFromCustomFile(self) -> None:
        customFilePath = self._settings.customFilePath.value

        if customFilePath is not None and customFilePath.is_file():
            self._object.read(customFilePath)
            self._customInitializer.setArray(self._object.getArray())

    def saveObject(self, filePath: Path) -> None:
        self._object.write(filePath)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setInitializerFromSettings()
        elif observable is self._reinitObservable:
            self._preloadObjectFromCustomFile()
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

    def getCurrentInitializer(self) -> str:
        return self._initializer.getInitializer()

    def setCurrentInitializer(self, name: str) -> None:
        self._initializer.setInitializer(name)

    def openObject(self, filePath: Path) -> None:
        self._initializer.openObject(filePath)

    def saveObject(self, filePath: Path) -> None:
        self._initializer.saveObject(filePath)

    def initializeObject(self) -> None:
        self._initializer.initializeObject()

    def getObject(self) -> numpy.ndarray:
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
