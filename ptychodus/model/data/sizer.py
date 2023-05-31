from __future__ import annotations

from ...api.data import DiffractionPatternData
from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ..detector import Detector
from .settings import DiffractionPatternSettings


class DiffractionPatternSizer(Observable, Observer):

    def __init__(self, settings: DiffractionPatternSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector
        self._sliceX = slice(0)
        self._sliceY = slice(0)

    @classmethod
    def createInstance(cls, settings: DiffractionPatternSettings,
                       detector: Detector) -> DiffractionPatternSizer:
        sizer = cls(settings, detector)
        sizer._updateSlicesAndNotifyObservers()
        settings.addObserver(sizer)
        detector.addObserver(sizer)
        return sizer

    def isCropEnabled(self) -> bool:
        return self._settings.cropEnabled.value

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.value = value

    def getExtentXLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsX())

    def getExtentXInPixels(self) -> int:
        limitsInPixels = self.getExtentXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropExtentXInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.upper

    def getCenterXLimitsInPixels(self) -> Interval[int]:
        lower = self.getExtentXInPixels() // 2
        upper = self._detector.getNumberOfPixelsX() - 1 - lower
        return Interval[int](lower, upper)

    def getCenterXInPixels(self) -> int:
        limitsInPixels = self.getCenterXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropCenterXInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.lower

    def getExtentYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsY())

    def getExtentYInPixels(self) -> int:
        limitsInPixels = self.getExtentYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropExtentYInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.upper

    def getCenterYLimitsInPixels(self) -> Interval[int]:
        lower = self.getExtentYInPixels() // 2
        upper = self._detector.getNumberOfPixelsY() - 1 - lower
        return Interval[int](lower, upper)

    def getCenterYInPixels(self) -> int:
        limitsInPixels = self.getCenterYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropCenterYInPixels.value) \
                if self.isCropEnabled() else limitsInPixels.lower

    def getExtentInPixels(self) -> ImageExtent:
        return ImageExtent(width=self.getExtentXInPixels(), height=self.getExtentYInPixels())

    def __call__(self, data: DiffractionPatternData) -> DiffractionPatternData:
        return data[:, self._sliceY, self._sliceX] if self.isCropEnabled() else data

    def _updateSlicesAndNotifyObservers(self) -> None:
        centerXInPixels = self.getCenterXInPixels()
        radiusXInPixels = self.getExtentXInPixels() // 2
        self._sliceX = slice(centerXInPixels - radiusXInPixels, centerXInPixels + radiusXInPixels)

        centerYInPixels = self.getCenterYInPixels()
        radiusYInPixels = self.getExtentYInPixels() // 2
        self._sliceY = slice(centerYInPixels - radiusYInPixels, centerYInPixels + radiusYInPixels)

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._updateSlicesAndNotifyObservers()
        elif observable is self._detector:
            self._updateSlicesAndNotifyObservers()
