from __future__ import annotations
from typing import Final

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

    def setNumberOfDataThreads(self, number: int) -> None:
        self._settings.numberOfDataThreads.value = number

    def getNumberOfDataThreads(self) -> int:
        return self._settings.numberOfDataThreads.value

    def getMinNumberOfDataThreads(self) -> int:
        return 1

    def getMaxNumberOfDataThreads(self) -> int:
        return 16

    def isCropEnabled(self) -> bool:
        return self._sizer.isCropEnabled()

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.value = value

    def getMinCropCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimitsInPixels().lower

    def getMaxCropCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimitsInPixels().upper

    def getCropCenterXInPixels(self) -> int:
        return self._sizer.getCenterXInPixels()

    def setCropCenterXInPixels(self, value: int) -> None:
        self._settings.cropCenterXInPixels.value = value

    def getMinCropCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimitsInPixels().lower

    def getMaxCropCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimitsInPixels().upper

    def getCropCenterYInPixels(self) -> int:
        return self._sizer.getCenterYInPixels()

    def setCropCenterYInPixels(self, value: int) -> None:
        self._settings.cropCenterYInPixels.value = value

    def getMinCropExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimitsInPixels().lower

    def getMaxCropExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimitsInPixels().upper

    def getCropExtentXInPixels(self) -> int:
        return self._sizer.getExtentXInPixels()

    def setCropExtentXInPixels(self, value: int) -> None:
        self._settings.cropExtentXInPixels.value = value

    def getMinCropExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimitsInPixels().lower

    def getMaxCropExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimitsInPixels().upper

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

    def getMinThreshold(self) -> int:
        return 0

    def getMaxThreshold(self) -> int:
        return self.MAX_INT

    def getThreshold(self) -> int:
        return self._settings.threshold.value

    def setThreshold(self, value: int) -> None:
        self._settings.threshold.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
