from __future__ import annotations

from ptychodus.api.geometry import ImageExtent, Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.patterns import DiffractionPatternArrayType

from .detector import Detector
from .settings import PatternSettings


class PatternSizer(Observable, Observer):
    def __init__(self, settings: PatternSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector
        self._sliceX = slice(0)
        self._sliceY = slice(0)

        self._updateSlicesAndNotifyObservers()
        settings.addObserver(self)
        detector.addObserver(self)

    def isCropEnabled(self) -> bool:
        return self._settings.cropEnabled.getValue()

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.setValue(value)

    def getWidthLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.widthInPixels.getValue())

    def getWidthInPixels(self) -> int:
        if self.isCropEnabled():
            limits = self.getWidthLimitsInPixels()
            return limits.clamp(self._settings.cropWidthInPixels.getValue())

        return self._detector.widthInPixels.getValue()

    def getCenterXLimitsInPixels(self) -> Interval[int]:
        return Interval[int](0, self._detector.widthInPixels.getValue())

    def getCenterXInPixels(self) -> int:
        limitsInPixels = self.getCenterXLimitsInPixels()
        return (
            limitsInPixels.clamp(self._settings.cropCenterXInPixels.getValue())
            if self.isCropEnabled()
            else limitsInPixels.midrange
        )

    def _getSafeCenterXInPixels(self) -> int:
        lower = self.getWidthInPixels() // 2
        upper = self._detector.widthInPixels.getValue() - 1 - lower
        limits = Interval[int](lower, upper)
        return limits.clamp(self.getCenterXInPixels())

    def getPixelWidthInMeters(self) -> float:
        return self._detector.pixelWidthInMeters.getValue()

    def getWidthInMeters(self) -> float:
        return self.getWidthInPixels() * self.getPixelWidthInMeters()

    def getHeightLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.heightInPixels.getValue())

    def getHeightInPixels(self) -> int:
        if self.isCropEnabled():
            limits = self.getHeightLimitsInPixels()
            return limits.clamp(self._settings.cropHeightInPixels.getValue())

        return self._detector.heightInPixels.getValue()

    def getCenterYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](0, self._detector.heightInPixels.getValue())

    def getCenterYInPixels(self) -> int:
        limitsInPixels = self.getCenterYLimitsInPixels()
        return (
            limitsInPixels.clamp(self._settings.cropCenterYInPixels.getValue())
            if self.isCropEnabled()
            else limitsInPixels.midrange
        )

    def _getSafeCenterYInPixels(self) -> int:
        lower = self.getHeightInPixels() // 2
        upper = self._detector.heightInPixels.getValue() - 1 - lower
        limits = Interval[int](lower, upper)
        return limits.clamp(self.getCenterYInPixels())

    def getPixelHeightInMeters(self) -> float:
        return self._detector.pixelHeightInMeters.getValue()

    def getHeightInMeters(self) -> float:
        return self.getHeightInPixels() * self.getPixelHeightInMeters()

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self.getWidthInPixels(),
            heightInPixels=self.getHeightInPixels(),
        )

    def __call__(self, data: DiffractionPatternArrayType) -> DiffractionPatternArrayType:
        return data[:, self._sliceY, self._sliceX] if self.isCropEnabled() else data

    def _updateSlicesAndNotifyObservers(self) -> None:
        centerXInPixels = self._getSafeCenterXInPixels()
        radiusXInPixels = self.getWidthInPixels() // 2
        self._sliceX = slice(centerXInPixels - radiusXInPixels, centerXInPixels + radiusXInPixels)

        centerYInPixels = self._getSafeCenterYInPixels()
        radiusYInPixels = self.getHeightInPixels() // 2
        self._sliceY = slice(centerYInPixels - radiusYInPixels, centerYInPixels + radiusYInPixels)

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._updateSlicesAndNotifyObservers()
        elif observable is self._detector:
            self._updateSlicesAndNotifyObservers()
