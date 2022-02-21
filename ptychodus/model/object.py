from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from typing import Callable, Tuple
import logging
import math

import numpy

from .detector import Detector
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
        self.customFilePath = settingsGroup.createPathEntry('CustomFilePath', Path())

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ObjectSettings:
        settings = cls(settingsRegistry.createGroup('Object'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class ObjectSizer(Observable, Observer):
    def __init__(self, scanSequence: ScanSequence, detector: Detector, probe: Probe) -> None:
        super().__init__()
        self._scanSequence = scanSequence
        self._detector = detector
        self._probe = probe
        self._scanExtentXInMeters = Decimal()
        self._scanExtentYInMeters = Decimal()

    @classmethod
    def createInstance(cls, scanSequence: ScanSequence, detector: Detector,
                       probe: Probe) -> ObjectSizer:
        sizer = cls(scanSequence, detector, probe)
        scanSequence.addObserver(sizer)
        detector.addObserver(sizer)
        probe.addObserver(sizer)
        sizer._updateBoundingBox()
        return sizer

    @property
    def shape(self) -> Tuple[int, int]:
        return self.extentYInPixels, self.extentXInPixels

    @property
    def extentXInPixels(self) -> int:
        objectPlanePixelSizeXInMeters = self._probe.wavelengthInMeters \
                * self._detector.distanceToObjectInMeters / self._detector.extentXInMeters
        numberOfInteriorPixels = self._scanExtentXInMeters / objectPlanePixelSizeXInMeters
        return int(math.ceil(numberOfInteriorPixels)) + self.numberOfPaddingPixels

    @property
    def extentYInPixels(self) -> int:
        objectPlanePixelSizeYInMeters = self._probe.wavelengthInMeters \
                * self._detector.distanceToObjectInMeters / self._detector.extentYInMeters
        numberOfInteriorPixels = self._scanExtentYInMeters / objectPlanePixelSizeYInMeters
        return int(math.ceil(numberOfInteriorPixels)) + self.numberOfPaddingPixels

    @property
    def numberOfPaddingPixels(self) -> int:
        return 2 * (self._probe.extentInPixels // 2)

    def _updateBoundingBox(self) -> None:
        pointIter = iter(self._scanSequence)

        try:
            point = next(pointIter)
        except StopIteration:
            self._scanExtentXInMeters = Decimal()
            self._scanExtentYInMeters = Decimal()
            return

        minX = maxX = point.x
        minY = maxY = point.y

        for point in pointIter:
            minX = min(minX, point.x)
            maxX = max(maxX, point.x)
            minY = min(minY, point.y)
            maxY = max(maxY, point.y)

        self._scanExtentXInMeters = maxX - minX
        self._scanExtentYInMeters = maxY - minY

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._scanSequence:
            self._updateBoundingBox()
        elif observable is self._detector:
            self.notifyObservers()
        elif observable is self._probe:
            self.notifyObservers()


class ObjectIO:
    FILE_FILTER = 'NumPy Binary Files (*.npy)'

    def write(self, filePath: Path, estimate: numpy.ndarray) -> None:
        numpy.save(filePath, estimate)

    def read(self, filePath: Path) -> numpy.ndarray:
        return numpy.load(filePath)


class UniformRandomObjectInitializer(Callable):
    def __init__(self, sizer: ObjectSizer, rng: numpy.random.Generator) -> None:
        super().__init__()
        self._sizer = sizer
        self._rng = rng

    def __call__(self) -> numpy.ndarray:
        size = self._sizer.shape
        magnitude = numpy.sqrt(self._rng.uniform(low=0., high=1., size=size))
        phase = self._rng.uniform(low=0., high=2. * numpy.pi, size=size)
        return magnitude * numpy.exp(1.j * phase)

    def __str__(self) -> str:
        return 'Random'


class CustomObjectInitializer(Callable):
    def __init__(self, sizer: ObjectSizer) -> None:
        super().__init__()
        self._sizer = sizer
        self._initialObject = numpy.zeros(sizer.shape, dtype=complex)

    def setInitialObject(self, initialObject: numpy.ndarray) -> None:
        if not numpy.iscomplexobj(initialObject):
            raise TypeError('Object must be a complex-valued ndarray')

        self._initialObject = initialObject

    def __call__(self) -> numpy.ndarray:
        return self._initialObject

    def __str__(self) -> str:
        return 'Custom'


class ObjectPresenter(Observable, Observer):
    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer,
                 initializerList: list[Callable]) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._initializerList = initializerList
        self._initializer = initializerList[0]
        self._estimate = numpy.zeros((0, 0), dtype=complex)
        self._objectIO = ObjectIO()

    @classmethod
    def createInstance(cls, rng: numpy.random.Generator, settings: ObjectSettings,
                       sizer: ObjectSizer) -> ObjectPresenter:
        initializerList = list()
        initializerList.append(UniformRandomObjectInitializer(sizer, rng))
        initializerList.append(CustomObjectInitializer(sizer))

        presenter = cls(settings, sizer, initializerList)
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
        self._settings.customFilePath.value = filePath
        initialObject = self._objectIO.read(filePath)
        self.setCurrentInitializer('Custom')
        self._initializer.setInitialObject(initialObject)
        self.initializeObject()

    def saveObject(self, filePath: Path) -> None:
        self._objectIO.write(filePath, self._estimate)

    def initializeObject(self) -> None:
        self._estimate = self._initializer()
        self.notifyObservers()

    def getObject(self) -> numpy.ndarray:
        return self._estimate

    def update(self, observable: Observable) -> None:
        if observable is self._settings.initializer:
            self.setCurrentInitializerFromSettings()
        elif observable is self._sizer:
            self.notifyObservers()
