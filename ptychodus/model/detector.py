from __future__ import annotations
from pathlib import Path
from decimal import Decimal

import numpy

from .image import ImageSequence
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup


class DetectorSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.dataPath = settingsGroup.createPathEntry('DataPath', None)
        self.numberOfPixelsX = settingsGroup.createIntegerEntry('NumberOfPixelsX', 1024)
        self.numberOfPixelsY = settingsGroup.createIntegerEntry('NumberOfPixelsY', 1024)
        self.pixelSizeXInMeters = settingsGroup.createRealEntry('PixelSizeXInMeters', '75e-6')
        self.pixelSizeYInMeters = settingsGroup.createRealEntry('PixelSizeYInMeters', '75e-6')
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '2')
        self.defocusDistanceInMeters = settingsGroup.createRealEntry('DefocusDistanceInMeters',
                                                                     '800e-6')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DetectorSettings:
        settings = cls(settingsRegistry.createGroup('Detector'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class Detector(Observable, Observer):
    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> Detector:
        detector = cls(settings)
        settings.detectorDistanceInMeters.addObserver(detector)
        settings.pixelSizeXInMeters.addObserver(detector)
        settings.pixelSizeYInMeters.addObserver(detector)
        return detector

    @property
    def numberOfPixelsX(self) -> int:
        return self._settings.numberOfPixelsX.value

    @property
    def pixelSizeXInMeters(self) -> Decimal:
        return self._settings.pixelSizeXInMeters.value

    @property
    def numberOfPixelsY(self) -> int:
        return self._settings.numberOfPixelsY.value

    @property
    def pixelSizeYInMeters(self) -> Decimal:
        return self._settings.pixelSizeYInMeters.value

    @property
    def distanceToObjectInMeters(self) -> Decimal:
        return self._settings.detectorDistanceInMeters.value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class DetectorPresenter(Observer, Observable):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> DetectorPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getMinNumberOfPixelsX(self) -> int:
        return 0

    def getMaxNumberOfPixelsX(self) -> int:
        return self.MAX_INT

    def getNumberOfPixelsX(self) -> int:
        return self._settings.numberOfPixelsX.value

    def setNumberOfPixelsX(self, value: int) -> None:
        self._settings.numberOfPixelsX.value = value

    def getMinNumberOfPixelsY(self) -> int:
        return 0

    def getMaxNumberOfPixelsY(self) -> int:
        return self.MAX_INT

    def getNumberOfPixelsY(self) -> int:
        return self._settings.numberOfPixelsY.value

    def setNumberOfPixelsY(self, value: int) -> None:
        self._settings.numberOfPixelsY.value = value

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._settings.pixelSizeXInMeters.value

    def setPixelSizeXInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeXInMeters.value = value

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._settings.pixelSizeYInMeters.value

    def setPixelSizeYInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeYInMeters.value = value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._settings.detectorDistanceInMeters.value

    def setDetectorDistanceInMeters(self, value: Decimal) -> None:
        self._settings.detectorDistanceInMeters.value = value

    def getDefocusDistanceInMeters(self) -> Decimal:
        return self._settings.defocusDistanceInMeters.value

    def setDefocusDistanceInMeters(self, value: Decimal) -> None:
        self._settings.defocusDistanceInMeters.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class DetectorImagePresenter(Observable, Observer):
    def __init__(self, imageSequence: ImageSequence) -> None:
        super().__init__()
        self._imageSequence = imageSequence

    @classmethod
    def createInstance(cls, imageSequence: ImageSequence) -> DetectorImagePresenter:
        presenter = cls(imageSequence)
        imageSequence.addObserver(presenter)
        return presenter

    def setCurrentDatasetIndex(self, index: int) -> None:
        self._imageSequence.setCurrentDatasetIndex(index)

    def getCurrentDatasetIndex(self) -> int:
        return self._imageSequence.getCurrentDatasetIndex()

    def getNumberOfImages(self) -> int:
        return len(self._imageSequence)

    def getImage(self, index: int) -> numpy.ndarray:
        try:
            return self._imageSequence[index]
        except IndexError:
            return numpy.empty((0,0))

    def update(self, observable: Observable) -> None:
        if observable is self._imageSequence:
            self.notifyObservers()
