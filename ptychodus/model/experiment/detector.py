from __future__ import annotations
from decimal import Decimal
from typing import Final

from ...api.apparatus import ImageExtent, PixelGeometry
from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class DetectorSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.numberOfPixelsX = settingsGroup.createIntegerEntry('NumberOfPixelsX', 1024)
        self.pixelSizeXInMeters = settingsGroup.createRealEntry('PixelSizeXInMeters', '75e-6')
        self.numberOfPixelsY = settingsGroup.createIntegerEntry('NumberOfPixelsY', 1024)
        self.pixelSizeYInMeters = settingsGroup.createRealEntry('PixelSizeYInMeters', '75e-6')
        self.bitDepth = settingsGroup.createIntegerEntry('BitDepth', 8)
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

    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> Detector:
        detector = cls(settings)
        settings.addObserver(detector)
        return detector

    def getExtentInPixels(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=max(0, self._settings.numberOfPixelsX.value),
            heightInPixels=max(0, self._settings.numberOfPixelsY.value),
        )

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=max(0., float(self._settings.pixelSizeXInMeters.value)),
            heightInMeters=max(0., float(self._settings.pixelSizeYInMeters.value)),
        )

    def getBitDepth(self) -> int:
        return max(1, self._settings.bitDepth.value)

    def getDetectorDistanceInMeters(self) -> float:
        return max(0., float(self._settings.detectorDistanceInMeters.value))

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class DetectorPresenter(Observable, Observer):  # FIXME change method names to match api/view
    MAX_INT: Final[int] = 0x7FFFFFFF

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
        return Interval[int](0, self.MAX_INT)

    def getNumberOfPixelsX(self) -> int:
        return self._detector.getExtentInPixels().widthInPixels

    def setNumberOfPixelsX(self, value: int) -> None:
        self._settings.numberOfPixelsX.value = value

    def getPixelSizeXInMeters(self) -> Decimal:
        return Decimal(repr(self._detector.getPixelGeometry().widthInMeters))

    def setPixelSizeXInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeXInMeters.value = value

    def getNumberOfPixelsYLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getNumberOfPixelsY(self) -> int:
        return self._detector.getExtentInPixels().heightInPixels

    def setNumberOfPixelsY(self, value: int) -> None:
        self._settings.numberOfPixelsY.value = value

    def getPixelSizeYInMeters(self) -> Decimal:
        return Decimal(repr(self._detector.getPixelGeometry().heightInMeters))

    def setPixelSizeYInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeYInMeters.value = value

    def getPixelGeometry(self) -> PixelGeometry:
        return self._detector.getPixelGeometry()

    def getBitDepthLimits(self) -> Interval[int]:
        return Interval[int](1, 64)

    def getBitDepth(self) -> int:
        limits = self.getBitDepthLimits()
        return limits.clamp(self._settings.bitDepth.value)

    def setBitDepth(self, value: int) -> None:
        self._settings.bitDepth.value = value

    def getDetectorDistanceInMeters(self) -> Decimal:
        # FIXME from_float -> repr
        return Decimal(repr(self._detector.getDetectorDistanceInMeters()))

    def setDetectorDistanceInMeters(self, value: Decimal) -> None:
        self._settings.detectorDistanceInMeters.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
