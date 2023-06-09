from __future__ import annotations
from decimal import Decimal
from typing import Final

from ..api.geometry import Interval
from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup


class DetectorSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.numberOfPixelsX = settingsGroup.createIntegerEntry('NumberOfPixelsX', 1024)
        self.pixelSizeXInMeters = settingsGroup.createRealEntry('PixelSizeXInMeters', '75e-6')
        self.numberOfPixelsY = settingsGroup.createIntegerEntry('NumberOfPixelsY', 1024)
        self.pixelSizeYInMeters = settingsGroup.createRealEntry('PixelSizeYInMeters', '75e-6')
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '2')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DetectorSettings:
        settings = cls(settingsRegistry.createGroup('Detector'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class Detector(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> Detector:
        detector = cls(settings)
        settings.addObserver(detector)
        return detector

    def getNumberOfPixelsXLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getNumberOfPixelsX(self) -> int:
        limits = self.getNumberOfPixelsXLimits()
        return limits.clamp(self._settings.numberOfPixelsX.value)

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._settings.pixelSizeXInMeters.value

    def getNumberOfPixelsYLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getNumberOfPixelsY(self) -> int:
        limits = self.getNumberOfPixelsYLimits()
        return limits.clamp(self._settings.numberOfPixelsY.value)

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._settings.pixelSizeYInMeters.value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._settings.detectorDistanceInMeters.value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class DetectorPresenter(Observable, Observer):

    def __init__(self, settings: DetectorSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector

    @classmethod
    def createInstance(cls, settings: DetectorSettings, detector: Detector) -> DetectorPresenter:
        presenter = cls(settings, detector)
        detector.addObserver(presenter)
        return presenter

    def getNumberOfPixelsXLimits(self) -> Interval[int]:
        return self._detector.getNumberOfPixelsXLimits()

    def getNumberOfPixelsX(self) -> int:
        return self._detector.getNumberOfPixelsX()

    def setNumberOfPixelsX(self, value: int) -> None:
        self._settings.numberOfPixelsX.value = value

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._detector.getPixelSizeXInMeters()

    def setPixelSizeXInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeXInMeters.value = value

    def getNumberOfPixelsYLimits(self) -> Interval[int]:
        return self._detector.getNumberOfPixelsYLimits()

    def getNumberOfPixelsY(self) -> int:
        return self._detector.getNumberOfPixelsY()

    def setNumberOfPixelsY(self, value: int) -> None:
        self._settings.numberOfPixelsY.value = value

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._detector.getPixelSizeYInMeters()

    def setPixelSizeYInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeYInMeters.value = value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._detector.getDetectorDistanceInMeters()

    def setDetectorDistanceInMeters(self, value: Decimal) -> None:
        self._settings.detectorDistanceInMeters.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
