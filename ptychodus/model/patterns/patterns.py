from __future__ import annotations
from typing import Final

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer

from .settings import PatternSettings
from .sizer import PatternSizer


class DiffractionPatternPresenter(Observable, Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, settings: PatternSettings, sizer: PatternSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: PatternSettings,
                       sizer: PatternSizer) -> DiffractionPatternPresenter:
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

    def getCropWidthLimitsInPixels(self) -> Interval[int]:
        return self._sizer.getWidthLimitsInPixels()

    def getCropWidthInPixels(self) -> int:
        return self._sizer.getWidthInPixels()

    def setCropWidthInPixels(self, value: int) -> None:
        self._settings.cropWidthInPixels.value = value

    def getCropHeightLimitsInPixels(self) -> Interval[int]:
        return self._sizer.getHeightLimitsInPixels()

    def getCropHeightInPixels(self) -> int:
        return self._sizer.getHeightInPixels()

    def setCropHeightInPixels(self, value: int) -> None:
        self._settings.cropHeightInPixels.value = value

    def isFlipXEnabled(self) -> bool:
        return self._settings.flipXEnabled.value

    def setFlipXEnabled(self, value: bool) -> None:
        self._settings.flipXEnabled.value = value

    def isFlipYEnabled(self) -> bool:
        return self._settings.flipYEnabled.value

    def setFlipYEnabled(self, value: bool) -> None:
        self._settings.flipYEnabled.value = value

    def isValueLowerBoundEnabled(self) -> bool:
        return self._settings.valueLowerBoundEnabled.value

    def setValueLowerBoundEnabled(self, value: bool) -> None:
        self._settings.valueLowerBoundEnabled.value = value

    def getValueLowerBoundLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getValueLowerBound(self) -> int:
        return self._settings.valueLowerBound.value

    def setValueLowerBound(self, value: int) -> None:
        self._settings.valueLowerBound.value = value

    def isValueUpperBoundEnabled(self) -> bool:
        return self._settings.valueUpperBoundEnabled.value

    def setValueUpperBoundEnabled(self, value: bool) -> None:
        self._settings.valueUpperBoundEnabled.value = value

    def getValueUpperBoundLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getValueUpperBound(self) -> int:
        return self._settings.valueUpperBound.value

    def setValueUpperBound(self, value: int) -> None:
        self._settings.valueUpperBound.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
