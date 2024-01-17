from __future__ import annotations

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.patterns import DiffractionPatternArrayType, ImageExtent
from .detector import Detector
from .settings import DiffractionPatternSettings


class PatternSizer(Observable, Observer):

    def __init__(self, settings: DiffractionPatternSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector
        self._sliceX = slice(0)
        self._sliceY = slice(0)

    @classmethod
    def createInstance(cls, settings: DiffractionPatternSettings,
                       detector: Detector) -> PatternSizer:
        sizer = cls(settings, detector)
        sizer._updateSlicesAndNotifyObservers()
        settings.addObserver(sizer)
        detector.addObserver(sizer)
        return sizer

    def isCropEnabled(self) -> bool:
        return self._settings.cropEnabled.value

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.value = value

    def getWidthLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getImageExtent().widthInPixels)

    def getWidthInPixels(self) -> int:
        limitsInPixels = self.getWidthLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropWidthInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.upper

    def getCenterXLimitsInPixels(self) -> Interval[int]:
        return Interval[int](0, self._detector.getImageExtent().widthInPixels)

    def getCenterXInPixels(self) -> int:
        limitsInPixels = self.getCenterXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropCenterXInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.midrange

    def _getSafeCenterXInPixels(self) -> int:
        lower = self.getWidthInPixels() // 2
        upper = self._detector.getImageExtent().widthInPixels - 1 - lower
        limits = Interval[int](lower, upper)
        return limits.clamp(self.getCenterXInPixels())

    def getPixelWidthInMeters(self) -> float:
        return float(self._detector.getPixelWidthInMeters())

    def getWidthInMeters(self) -> float:
        return self.getWidthInPixels() * self.getPixelWidthInMeters()  # FIXME safe?

    def getHeightLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getImageExtent().heightInPixels)

    def getHeightInPixels(self) -> int:
        limitsInPixels = self.getHeightLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropHeightInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.upper

    def getCenterYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](0, self._detector.getImageExtent().heightInPixels)

    def getCenterYInPixels(self) -> int:
        limitsInPixels = self.getCenterYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropCenterYInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.midrange

    def _getSafeCenterYInPixels(self) -> int:
        lower = self.getHeightInPixels() // 2
        upper = self._detector.getImageExtent().heightInPixels - 1 - lower
        limits = Interval[int](lower, upper)
        return limits.clamp(self.getCenterYInPixels())

    def getPixelHeightInMeters(self) -> float:
        return float(self._detector.getPixelHeightInMeters())

    def getHeightInMeters(self) -> float:
        return self.getHeightInPixels() * self.getPixelHeightInMeters()  # FIXME safe?

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
