from __future__ import annotations
from decimal import Decimal
from typing import Final

from ptychodus.api.geometry import ImageExtent, Interval, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import DecimalParameter, IntegerParameter
from ptychodus.api.settings import SettingsRegistry


class Detector(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Detector")
        self._settingsGroup.addObserver(self)

        self._widthInPixels = IntegerParameter(self._settingsGroup, "WidthInPixels", 1024)
        self._pixelWidthInMeters = DecimalParameter(
            self._settingsGroup, "PixelWidthInMeters", "75e-6"
        )
        self._heightInPixels = IntegerParameter(self._settingsGroup, "HeightInPixels", 1024)
        self._pixelHeightInMeters = DecimalParameter(
            self._settingsGroup, "PixelHeightInMeters", "75e-6"
        )
        self._bitDepth = IntegerParameter(self._settingsGroup, "BitDepth", 8)

    def getWidthInPixels(self) -> int:
        return max(0, self._widthInPixels.getValue())

    def setWidthInPixels(self, widthInPixels: int) -> None:
        self._widthInPixels.setValue(widthInPixels)

    def getHeightInPixels(self) -> int:
        return max(0, self._heightInPixels.getValue())

    def setHeightInPixels(self, heightInPixels: int) -> None:
        self._heightInPixels.setValue(heightInPixels)

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self.getWidthInPixels(),
            heightInPixels=self.getHeightInPixels(),
        )

    def setImageExtent(self, imageExtent: ImageExtent) -> None:
        self.setWidthInPixels(imageExtent.widthInPixels)
        self.setHeightInPixels(imageExtent.heightInPixels)

    def getPixelWidthInMeters(self) -> Decimal:
        return max(Decimal(), self._pixelWidthInMeters.getValue())

    def setPixelWidthInMeters(self, pixelWidthInMeters: Decimal) -> None:
        self._pixelWidthInMeters.setValue(pixelWidthInMeters)

    def getPixelHeightInMeters(self) -> Decimal:
        return max(Decimal(), self._pixelHeightInMeters.getValue())

    def setPixelHeightInMeters(self, pixelHeightInMeters: Decimal) -> None:
        self._pixelHeightInMeters.setValue(pixelHeightInMeters)

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=float(self.getPixelWidthInMeters()),
            heightInMeters=float(self.getPixelHeightInMeters()),
        )

    def setPixelGeometry(self, pixelGeometry: PixelGeometry) -> None:
        self.setPixelWidthInMeters(Decimal(repr(pixelGeometry.widthInMeters)))
        self.setPixelHeightInMeters(Decimal(repr(pixelGeometry.heightInMeters)))

    def getBitDepth(self) -> int:
        return max(1, self._bitDepth.getValue())

    def setBitDepth(self, bitDepth: int) -> None:
        self._bitDepth.setValue(bitDepth)

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
        return self._detector.getWidthInPixels()

    def setWidthInPixels(self, value: int) -> None:
        self._detector.setWidthInPixels(value)

    def getPixelWidthInMeters(self) -> Decimal:
        return self._detector.getPixelWidthInMeters()

    def setPixelWidthInMeters(self, value: Decimal) -> None:
        self._detector.setPixelWidthInMeters(value)

    def getHeightInPixelsLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getHeightInPixels(self) -> int:
        return self._detector.getHeightInPixels()

    def setHeightInPixels(self, value: int) -> None:
        self._detector.setHeightInPixels(value)

    def getPixelHeightInMeters(self) -> Decimal:
        return self._detector.getPixelHeightInMeters()

    def setPixelHeightInMeters(self, value: Decimal) -> None:
        self._detector.setPixelHeightInMeters(value)

    def getPixelGeometry(self) -> PixelGeometry:
        return self._detector.getPixelGeometry()

    def getBitDepthLimits(self) -> Interval[int]:
        return Interval[int](1, 64)

    def getBitDepth(self) -> int:
        return self._detector.getBitDepth()

    def setBitDepth(self, value: int) -> None:
        self._detector.setBitDepth(value)

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
