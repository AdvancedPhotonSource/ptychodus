from __future__ import annotations
from typing import Final

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from .crop import CropSizer
from .settings import DiffractionPatternSettings


class DiffractionPatternPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: DiffractionPatternSettings, sizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: DiffractionPatternSettings,
                       sizer: CropSizer) -> DiffractionPatternPresenter:
        presenter = cls(settings, sizer)
        sizer.addObserver(presenter)
        return presenter

    def isCropEnabled(self) -> bool:
        return self._sizer.isCropEnabled()

    def setCropEnabled(self, value: bool) -> None:
        self._sizer.setCropEnabled(value)

    def getCropCenterXLimitsInPixels(self) -> Interval[int]:
        return self._sizer.getCenterXLimitsInPixels()

    def getCropCenterXInPixels(self) -> int:
        return self._sizer.getCenterXInPixels()

    def setCropCenterXInPixels(self, value: int) -> None:
        self._settings.cropCenterXInPixels.value = value

    def getCropCenterYLimitsInPixels(self) -> Interval[int]:
        return self._sizer.getCenterYLimitsInPixels()

    def getCropCenterYInPixels(self) -> int:
        return self._sizer.getCenterYInPixels()

    def setCropCenterYInPixels(self, value: int) -> None:
        self._settings.cropCenterYInPixels.value = value

    def getCropExtentXLimitsInPixels(self) -> Interval[int]:
        return self._sizer.getExtentXLimitsInPixels()

    def getCropExtentXInPixels(self) -> int:
        return self._sizer.getExtentXInPixels()

    def setCropExtentXInPixels(self, value: int) -> None:
        self._settings.cropExtentXInPixels.value = value

    def getCropExtentYLimitsInPixels(self) -> Interval[int]:
        return self._sizer.getExtentYLimitsInPixels()

    def getCropExtentYInPixels(self) -> int:
        return self._sizer.getExtentYInPixels()

    def setCropExtentYInPixels(self, value: int) -> None:
        self._settings.cropExtentYInPixels.value = value

    def isFlipXEnabled(self) -> bool:
        return self._settings.flipXEnabled.value

    def setFlipXEnabled(self, value: bool) -> None:
        self._settings.flipXEnabled.value = value

    def isFlipYEnabled(self) -> bool:
        return self._settings.flipYEnabled.value

    def setFlipYEnabled(self, value: bool) -> None:
        self._settings.flipYEnabled.value = value

    def isThresholdEnabled(self) -> bool:
        return self._settings.thresholdEnabled.value

    def setThresholdEnabled(self, value: bool) -> None:
        self._settings.thresholdEnabled.value = value

    def getThresholdValueLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getThresholdValue(self) -> int:
        return self._settings.thresholdValue.value

    def setThresholdValue(self, value: int) -> None:
        self._settings.thresholdValue.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
