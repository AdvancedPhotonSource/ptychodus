from __future__ import annotations

import numpy

from .detector import Detector
from .geometry import Interval
from .image import ImageSequence
from .observer import Observable, Observer
from .settings import SettingsGroup, SettingsRegistry


class CropSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.cropEnabled = settingsGroup.createBooleanEntry('CropEnabled', False)
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


class CropSizer(Observer, Observable):
    MAX_INT = 0x7FFFFFFF

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

    def getExtentXLimits(self) -> Interval[int]:
        return Interval[int](1, self._detector.numberOfPixelsX)

    def getExtentX(self) -> int:
        limits = self.getExtentXLimits()
        return limits.clamp(self._settings.extentXInPixels.value)

    def getCenterXLimits(self) -> Interval[int]:
        radius = self.getExtentX() // 2
        return Interval[int](radius, self._detector.numberOfPixelsX - 1 - radius)

    def getCenterX(self) -> int:
        limits = self.getCenterXLimits()
        return limits.clamp(self._settings.centerXInPixels.value)

    def getSliceX(self) -> slice:
        center = self.getCenterX()
        radius = self.getExtentX() // 2
        return slice(center - radius, center + radius + 1)

    def getExtentYLimits(self) -> Interval[int]:
        return Interval[int](1, self._detector.numberOfPixelsY)

    def getExtentY(self) -> int:
        limits = self.getExtentYLimits()
        return limits.clamp(self._settings.extentYInPixels.value)

    def getCenterYLimits(self) -> Interval[int]:
        radius = self.getExtentY() // 2
        return Interval[int](radius, self._detector.numberOfPixelsY - 1 - radius)

    def getCenterY(self) -> int:
        limits = self.getCenterYLimits()
        return limits.clamp(self._settings.centerYInPixels.value)

    def getSliceY(self) -> slice:
        center = self.getCenterY()
        radius = self.getExtentY() // 2
        return slice(center - radius, center + radius + 1)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class CroppedImageSequence(ImageSequence):
    def __init__(self, sizer: CropSizer, imageSequence: ImageSequence) -> None:
        super().__init__()
        self._sizer = sizer
        self._imageSequence = imageSequence

    @classmethod
    def createInstance(cls, sizer: CropSizer,
                       imageSequence: ImageSequence) -> CroppedImageSequence:
        croppedImageSequence = cls(sizer, imageSequence)
        sizer.addObserver(croppedImageSequence)
        imageSequence.addObserver(croppedImageSequence)
        return croppedImageSequence

    def setCurrentDatasetIndex(self, index: int) -> None:
        self._imageSequence.setCurrentDatasetIndex(index)

    def getCurrentDatasetIndex(self) -> int:
        return self._imageSequence.getCurrentDatasetIndex()

    def getWidth(self) -> int:
        return self._imageSequence.getWidth()

    def getHeight(self) -> int:
        return self._imageSequence.getHeight()

    def __getitem__(self, index: int) -> numpy.ndarray:
        img = self._imageSequence[index]

        if self._sizer.isCropEnabled():
            sliceX = self._sizer.getSliceX()
            sliceY = self._sizer.getSliceY()
            img = numpy.copy(img[sliceY, sliceX])

        return img

    def __len__(self) -> int:
        return len(self._imageSequence)

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._imageSequence:
            self.notifyObservers()


class CropPresenter(Observer, Observable):
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
        return self._sizer.getCenterXLimits().lower

    def getMaxCenterXInPixels(self) -> int:
        return self._sizer.getCenterXLimits().upper

    def getCenterXInPixels(self) -> int:
        return self._sizer.getCenterX()

    def setCenterXInPixels(self, value: int) -> None:
        self._settings.centerXInPixels.value = value

    def getMinCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimits().lower

    def getMaxCenterYInPixels(self) -> int:
        return self._sizer.getCenterYLimits().upper

    def getCenterYInPixels(self) -> int:
        return self._sizer.getCenterY()

    def setCenterYInPixels(self, value: int) -> None:
        self._settings.centerYInPixels.value = value

    def getMinExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimits().lower

    def getMaxExtentXInPixels(self) -> int:
        return self._sizer.getExtentXLimits().upper

    def getExtentXInPixels(self) -> int:
        return self._sizer.getExtentX()

    def setExtentXInPixels(self, value: int) -> None:
        self._settings.extentXInPixels.value = value

    def getMinExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimits().lower

    def getMaxExtentYInPixels(self) -> int:
        return self._sizer.getExtentYLimits().upper

    def getExtentYInPixels(self) -> int:
        return self._sizer.getExtentY()

    def setExtentYInPixels(self, value: int) -> None:
        self._settings.extentYInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
