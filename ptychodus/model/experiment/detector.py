from __future__ import annotations
from decimal import Decimal
from typing import Final

from ...api.apparatus import ImageExtent, PixelGeometry
from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class Detector(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.widthInPixels = settingsGroup.createIntegerEntry('WidthInPixels', 1024)
        self.pixelWidthInMeters = settingsGroup.createRealEntry('PixelWidthInMeters', '75e-6')
        self.heightInPixels = settingsGroup.createIntegerEntry('HeightInPixels', 1024)
        self.pixelHeightInMeters = settingsGroup.createRealEntry('PixelHeightInMeters', '75e-6')
        self.bitDepth = settingsGroup.createIntegerEntry('BitDepth', 8)
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '2')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> Detector:
        settingsGroup = settingsRegistry.createGroup('Detector')
        detector = cls(settingsGroup)
        settingsGroup.addObserver(detector)
        return detector

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=max(0, self.widthInPixels.value),
            heightInPixels=max(0, self.heightInPixels.value),
        )

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=max(0., float(self.pixelWidthInMeters.value)),
            heightInMeters=max(0., float(self.pixelHeightInMeters.value)),
        )

    def getBitDepth(self) -> int:
        return max(1, self.bitDepth.value)

    def getDetectorDistanceInMeters(self) -> float:
        return max(0., float(self.detectorDistanceInMeters.value))

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class DetectorPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, detector: Detector) -> None:
        super().__init__()
        self._detector = detector

    @classmethod
    def createInstance(cls, detector: Detector) -> DetectorPresenter:
        presenter = cls(detector)
        detector.addObserver(presenter)
        return presenter

    def getWidthInPixelsLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getWidthInPixels(self) -> int:
        return self._detector.widthInPixels.value

    def setWidthInPixels(self, value: int) -> None:
        self._detector.widthInPixels.value = value

    def getPixelWidthInMeters(self) -> Decimal:
        return self._detector.pixelWidthInMeters.value

    def setPixelWidthInMeters(self, value: Decimal) -> None:
        self._detector.pixelWidthInMeters.value = value

    def getHeightInPixelsLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getHeightInPixels(self) -> int:
        return self._detector.heightInPixels.value

    def setHeightInPixels(self, value: int) -> None:
        self._detector.heightInPixels.value = value

    def getPixelHeightInMeters(self) -> Decimal:
        return self._detector.pixelHeightInMeters.value

    def setPixelHeightInMeters(self, value: Decimal) -> None:
        self._detector.pixelHeightInMeters.value = value

    def getPixelGeometry(self) -> PixelGeometry:
        return self._detector.getPixelGeometry()

    def getBitDepthLimits(self) -> Interval[int]:
        return Interval[int](1, 64)

    def getBitDepth(self) -> int:
        limits = self.getBitDepthLimits()
        return limits.clamp(self._detector.bitDepth.value)

    def setBitDepth(self, value: int) -> None:
        self._detector.bitDepth.value = value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._detector.detectorDistanceInMeters.value

    def setDetectorDistanceInMeters(self, value: Decimal) -> None:
        self._detector.detectorDistanceInMeters.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
