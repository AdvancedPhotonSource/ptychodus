from __future__ import annotations

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup
from ..detector import Detector


class CropSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.cropEnabled = settingsGroup.createBooleanEntry('CropEnabled', True)
        self.centerXInPixels = settingsGroup.createIntegerEntry('CenterXInPixels', 32)
        self.centerYInPixels = settingsGroup.createIntegerEntry('CenterYInPixels', 32)
        self.extentXInPixels = settingsGroup.createIntegerEntry('ExtentXInPixels', 64)
        self.extentYInPixels = settingsGroup.createIntegerEntry('ExtentYInPixels', 64)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> CropSettings:
        settings = cls(settingsRegistry.createGroup('Crop'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class CropSizer(Observable, Observer):

    def __init__(self, settings: CropSettings, detector: Detector) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector

    @classmethod
    def createInstance(cls, settings: CropSettings, detector: Detector) -> CropSizer:
        sizer = cls(settings, detector)
        settings.addObserver(sizer)
        detector.addObserver(sizer)
        return sizer

    def isCropEnabled(self) -> bool:
        return self._settings.cropEnabled.value

    def getExtentXLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsX())

    def getExtentXInPixels(self) -> int:
        limitsInPixels = self.getExtentXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.extentXInPixels.value)

    def getCenterXLimitsInPixels(self) -> Interval[int]:
        radiusInPixels = self.getExtentXInPixels() // 2
        return Interval[int](radiusInPixels,
                             self._detector.getNumberOfPixelsX() - 1 - radiusInPixels)

    def getCenterXInPixels(self) -> int:
        limitsInPixels = self.getCenterXLimitsInPixels()
        return limitsInPixels.clamp(self._settings.centerXInPixels.value)

    def getSliceX(self) -> slice:
        centerInPixels = self.getCenterXInPixels()
        radiusInPixels = self.getExtentXInPixels() // 2
        return slice(centerInPixels - radiusInPixels, centerInPixels + radiusInPixels)

    def getExtentYLimitsInPixels(self) -> Interval[int]:
        return Interval[int](1, self._detector.getNumberOfPixelsY())

    def getExtentYInPixels(self) -> int:
        limitsInPixels = self.getExtentYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.extentYInPixels.value)

    def getCenterYLimitsInPixels(self) -> Interval[int]:
        radiusInPixels = self.getExtentYInPixels() // 2
        return Interval[int](radiusInPixels,
                             self._detector.getNumberOfPixelsY() - 1 - radiusInPixels)

    def getCenterYInPixels(self) -> int:
        limitsInPixels = self.getCenterYLimitsInPixels()
        return limitsInPixels.clamp(self._settings.centerYInPixels.value)

    def getSliceY(self) -> slice:
        centerInPixels = self.getCenterYInPixels()
        radiusInPixels = self.getExtentYInPixels() // 2
        return slice(centerInPixels - radiusInPixels, centerInPixels + radiusInPixels)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._detector:
            self.notifyObservers()


class CropPresenter(Observable, Observer):

    def __init__(self, settings: CropSettings, sizer: CropSizer) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: CropSettings, sizer: CropSizer) -> CropPresenter:
        presenter = cls(settings, sizer)
        sizer.addObserver(presenter)
        return presenter

    def isCropEnabled(self) -> bool:
        return self._sizer.isCropEnabled()

    def setCropEnabled(self, value: bool) -> None:
        self._settings.cropEnabled.value = value

    def getMinCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimitsInPixels().lower

    def getMaxCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimitsInPixels().upper

    def getCenterXInPixels(self) -> int:
        return self._sizer.getCenterXInPixels()

    def setCenterXInPixels(self, value: int) -> None:
        self._settings.centerXInPixels.value = value

    def getMinCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimitsInPixels().lower

    def getMaxCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimitsInPixels().upper

    def getCenterYInPixels(self) -> int:
        return self._sizer.getCenterYInPixels()

    def setCenterYInPixels(self, value: int) -> None:
        self._settings.centerYInPixels.value = value

    def getMinExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimitsInPixels().lower

    def getMaxExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimitsInPixels().upper

    def getExtentXInPixels(self) -> int:
        return self._sizer.getExtentXInPixels()

    def setExtentXInPixels(self, value: int) -> None:
        self._settings.extentXInPixels.value = value

    def getMinExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimitsInPixels().lower

    def getMaxExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimitsInPixels().upper

    def getExtentYInPixels(self) -> int:
        return self._sizer.getExtentYInPixels()

    def setExtentYInPixels(self, value: int) -> None:
        self._settings.extentYInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
