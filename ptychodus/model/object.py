from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Callable
import logging

import numpy

from .geometry import Box, Interval
from .detector import Detector
from .image import ImageExtent
from .observer import Observable, Observer
from .probe import Probe
from .scan import ScanSequence
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
    def __init__(self, scanSequence: ScanSequence, detector: Detector,
                 probeSizer: ProbeSizer) -> None:
        super().__init__()
        self._scanSequence = scanSequence
        self._detector = detector
        self._probeSizer = probeSizer
        self._scanBoundingBoxInMeters = self._scanSequence.getBoundingBox()

    @classmethod
    def createInstance(cls, scanSequence: ScanSequence, detector: Detector,
                       probeSizer: ProbeSizer) -> ObjectSizer:
        sizer = cls(scanSequence, detector, probeSizer)
        scanSequence.addObserver(sizer)
        detector.addObserver(sizer)
        probeSizer.addObserver(sizer)
        return sizer

    @property
    def objectPlanePixelShapeInMeters(self) -> Tuple[Decimal, Decimal]:
        lambdaZ_m2 = self._probeSizer.getWavelengthInMeters(
        ) * self._detector.distanceToObjectInMeters
        px_m = lambdaZ_m2 / self._detector.extentXInMeters
        py_m = lambdaZ_m2 / self._detector.extentYInMeters
        return py_m, px_m

    def getScanBoundingBoxInMeters(self) -> Box[Decimal]:
        return self._scanBoundingBoxInMeters

    def getScanBoundingBoxInPixels(self) -> Box[Decimal]:
        assert len(self._scanBoundingBoxInMeters) == 2
        py_m, px_m = self.objectPlanePixelShapeInMeters
        ix = Interval(lower=self._scanBoundingBoxInMeters[0].lower / px_m,
                      upper=self._scanBoundingBoxInMeters[0].upper / px_m)
        iy = Interval(lower=self._scanBoundingBoxInMeters[1].lower / py_m,
                      upper=self._scanBoundingBoxInMeters[1].upper / py_m)
        return Box([ix, iy])

    def getScanExtent(self) -> ImageExtent:
        bbox_px = self.getScanBoundingBoxInPixels()
        scanWidth_px = int(numpy.ceil(bbox_px[0].length))
        scanHeight_px = int(numpy.ceil(bbox_px[1].length))
        return ImageExtent(width=scanWidth_px, height=scanHeight_px)

    def getPaddingExtent(self) -> ImageExtent:
        pad = 2 * (self._probeSizer.getProbeSize() // 2)
        return ImageExtent(width=pad, height=pad)

    def getObjectExtent(self) -> ImageExtent:
        return self.getScanExtent() + self.getPaddingExtent()

    def _updateBoundingBox(self) -> None:
        self._scanBoundingBoxInMeters = self._scanSequence.getBoundingBox()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._scanSequence:
            self._updateBoundingBox()
        elif observable is self._detector:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()


class Object(Observable):
    FILE_FILTER = 'NumPy Binary Files (*.npy)'

    def __init__(self, settings: ObjectSettings) -> None:
        super().__init__()
        self._settings = settings
        self._array = numpy.zeros((0, 0), dtype=complex)

    @classmethod
    def createInstance(cls, settings: ObjectSettings) -> Object:
        return cls(settings)

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


class UniformRandomObjectInitializer:
    def __init__(self, sizer: ObjectSizer, rng: numpy.random.Generator) -> None:
        super().__init__()
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
        super().__init__()
        self._sizer = sizer
        self._initialObject = numpy.zeros(sizer.getObjectExtent().shape, dtype=complex)

    def setInitialObject(self, initialObject: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(initialObject):
            raise TypeError('Object must be a complex-valued ndarray')

        self._initialObject = initialObject

    def __call__(self) -> numpy.ndarray:
        return self._initialObject

    def __str__(self) -> str:
        return 'Custom'


class ObjectPresenter(Observable, Observer):
    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer, obj: Object,
                 initializerList: Sequence[Callable]) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._object = obj
        self._initializerList = initializerList
        self._initializer = initializerList[0]

    @classmethod
    def createInstance(cls, rng: numpy.random.Generator, settings: ObjectSettings,
                       sizer: ObjectSizer, obj: Object) -> ObjectPresenter:
        initializerList = list()
        initializerList.append(UniformRandomObjectInitializer(sizer, rng))
        initializerList.append(CustomObjectInitializer(sizer))

        presenter = cls(settings, sizer, obj, initializerList)
        presenter.setCurrentInitializerFromSettings()
        settings.initializer.addObserver(presenter)
        sizer.addObserver(presenter)

        return presenter

    def getInitializerList(self) -> list[str]:
        return [str(initializer) for initializer in self._initializerList]

    def getCurrentInitializer(self) -> str:
        return str(self._initializer)

    def setCurrentInitializer(self, name: str) -> None:
        try:
            initializer = next(ini for ini in self._initializerList
                               if name.casefold() == str(ini).casefold())
        except StopIteration:
            return

        if initializer is not self._initializer:
            self._initializer = initializer
            self._settings.initializer.value = str(self._initializer)
            self.notifyObservers()

    def setCurrentInitializerFromSettings(self) -> None:
        self.setCurrentInitializer(self._settings.initializer.value)

    def openObject(self, filePath: Path) -> None:
        self._object.read(filePath)
        self.setCurrentInitializer('Custom')
        self._initializer.cacheObject(self._object.getArray())
        self.notifyObservers()

    def saveObject(self, filePath: Path) -> None:
        self._object.write(filePath)

    def initializeObject(self) -> None:
        self._object.setArray(self._initializer())
        self.notifyObservers()

    def getObject(self) -> numpy.ndarray:
        return self._object.getArray()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setCurrentInitializerFromSettings()
        elif observable is self._sizer:
            self.notifyObservers()
