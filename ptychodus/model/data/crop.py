from __future__ import annotations

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup
from ..detector import Detector
from .settings import DiffractionPatternSettings


class CropSizer(Observable, Observer):

    def __init__(self, settings: DiffractionPatternSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector

    @classmethod
    def createInstance(cls, settings: DiffractionPatternSettings, detector: Detector) -> CropSizer:
        sizer = cls(settings, detector)
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
        return limitsInPixels.clamp(self._settings.cropExtentXInPixels.value)

    def getCenterXLimitsInPixels(self) -> Interval[int]:
        radiusInPixels = self.getExtentXInPixels() // 2
        return Interval[int](radiusInPixels,
                             self._detector.getNumberOfPixelsX() - 1 - radiusInPixels)

    def getCenterXInPixels(self) -> int:
        limitsInPixels = self.getCenterXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropCenterXInPixels.value)

    def getSliceX(self) -> slice:
        centerInPixels = self.getCenterXInPixels()
        radiusInPixels = self.getExtentXInPixels() // 2
        return slice(centerInPixels - radiusInPixels, centerInPixels + radiusInPixels)

    def getExtentYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsY())

    def getExtentYInPixels(self) -> int:
        limitsInPixels = self.getExtentYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropExtentYInPixels.value)

    def getCenterYLimitsInPixels(self) -> Interval[int]:
        radiusInPixels = self.getExtentYInPixels() // 2
        return Interval[int](radiusInPixels,
                             self._detector.getNumberOfPixelsY() - 1 - radiusInPixels)

    def getCenterYInPixels(self) -> int:
        limitsInPixels = self.getCenterYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.cropCenterYInPixels.value)

    def getSliceY(self) -> slice:
        centerInPixels = self.getCenterYInPixels()
        radiusInPixels = self.getExtentYInPixels() // 2
        return slice(centerInPixels - radiusInPixels, centerInPixels + radiusInPixels)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._detector:
            self.notifyObservers()
