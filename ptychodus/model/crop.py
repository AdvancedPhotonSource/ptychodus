from __future__ import annotations
from decimal import Decimal

import numpy

from .detector import Detector
from .geometry import Interval
from .image import ImageSequence
from ..api.observer import Observable, Observer
from ..api.settings import SettingsGroup, SettingsRegistry


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


class CropSizer(Observer, Observable):
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

    def getExtentXInMeters(self) -> Decimal:
        return self.getExtentX() * self._detector.pixelSizeXInMeters

    def getCenterXLimits(self) -> Interval[int]:
        radius = self.getExtentX() // 2
        return Interval[int](radius, self._detector.numberOfPixelsX - 1 - radius)

    def getCenterX(self) -> int:
        limits = self.getCenterXLimits()
        return limits.clamp(self._settings.centerXInPixels.value)

    def getSliceX(self) -> slice:
        center = self.getCenterX()
        radius = self.getExtentX() // 2
        return slice(center - radius, center + radius)

    def getExtentYLimits(self) -> Interval[int]:
        return Interval[int](1, self._detector.numberOfPixelsY)

    def getExtentY(self) -> int:
        limits = self.getExtentYLimits()
        return limits.clamp(self._settings.extentYInPixels.value)

    def getExtentYInMeters(self) -> Decimal:
        return self.getExtentY() * self._detector.pixelSizeYInMeters

    def getCenterYLimits(self) -> Interval[int]:
        radius = self.getExtentY() // 2
        return Interval[int](radius, self._detector.numberOfPixelsY - 1 - radius)

    def getCenterY(self) -> int:
        limits = self.getCenterYLimits()
        return limits.clamp(self._settings.centerYInPixels.value)

    def getSliceY(self) -> slice:
        center = self.getCenterY()
        radius = self.getExtentY() // 2
        return slice(center - radius, center + radius)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class CroppedDiffractionDataset(DiffractionDataset):
    def __init__(self, sizer: CropSizer, dataset: DiffractionDataset) -> None:
        super().__init__()
        self._sizer = sizer
        self._dataset = dataset

    @classmethod
    def createInstance(cls, sizer: CropSizer,
                       dataset: DiffractionDataset) -> CroppedDiffractionDataset:
        croppedDataset = cls(sizer, dataset)
        sizer.addObserver(croppedDataset)
        dataset.addObserver(croppedDataset)
        return croppedDataset

    def __getitem__(self, index: int) -> numpy.ndarray:
        img = self._dataset[index]

        if self._sizer.isCropEnabled():
            sliceX = self._sizer.getSliceX()
            sliceY = self._sizer.getSliceY()
            img = numpy.copy(img[sliceY, sliceX])

        return img

    def __len__(self) -> int:
        return len(self._dataset)

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self.notifyObservers()
        elif observable is self._dataset:
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
