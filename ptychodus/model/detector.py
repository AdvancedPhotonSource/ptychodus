from __future__ import annotations
from pathlib import Path
from decimal import Decimal

import numpy

from .image import ImageSequence, CropSizer
from .observer import Observable, Observer
from .settings import SettingsRegistry, SettingsGroup


class DetectorSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.dataPath = settingsGroup.createPathEntry('DataPath', None)
        self.pixelSizeXInMeters = settingsGroup.createRealEntry('PixelSizeXInMeters', '75e-6')
        self.pixelSizeYInMeters = settingsGroup.createRealEntry('PixelSizeYInMeters', '75e-6')
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '2')
        self.defocusDistanceInMeters = settingsGroup.createRealEntry('DefocusDistanceInMeters',
                                                                     '800e-6')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> DetectorSettings:
        settings = cls(settingsRegistry.createGroup('Detector'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class Detector(Observable, Observer):
    def __init__(self, detectorSettings: DetectorSettings, cropSizer: CropSizer) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._cropSizer = cropSizer

    @classmethod
    def createInstance(cls, detectorSettings: DetectorSettings,
                       cropSizer: CropSizer) -> Detector:
        detector = cls(detectorSettings, cropSizer)
        detectorSettings.detectorDistanceInMeters.addObserver(detector)
        detectorSettings.pixelSizeXInMeters.addObserver(detector)
        detectorSettings.pixelSizeYInMeters.addObserver(detector)
        cropSizer.addObserver(detector)
        return detector

    @property
    def distanceToObjectInMeters(self) -> Decimal:
        return self._detectorSettings.detectorDistanceInMeters.value

    @property
    def extentXInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeXInMeters.value \
                * self._cropSizer.getExtentX()

    @property
    def extentYInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeYInMeters.value \
                * self._cropSizer.getExtentY()

    def update(self, observable: Observable) -> None:
        if observable is self._detectorSettings.detectorDistanceInMeters:
            self.notifyObservers()
        elif observable is self._detectorSettings.pixelSizeXInMeters:
            self.notifyObservers()
        elif observable is self._detectorSettings.pixelSizeYInMeters:
            self.notifyObservers()
        elif observable is self._cropSizer:
            self.notifyObservers()


class DetectorPresenter(Observer, Observable):
    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> DetectorPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._settings.pixelSizeXInMeters.value

    def setPixelSizeXInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeXInMeters.value = value

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._settings.pixelSizeYInMeters.value

    def setPixelSizeYInMeters(self, value: Decimal) -> None:
        self._settings.pixelSizeYInMeters.value = value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._settings.detectorDistanceInMeters.value

    def setDetectorDistanceInMeters(self, value: Decimal) -> None:
        self._settings.detectorDistanceInMeters.value = value

    def getDefocusDistanceInMeters(self) -> Decimal:
        return self._settings.defocusDistanceInMeters.value

    def setDefocusDistanceInMeters(self, value: Decimal) -> None:
        self._settings.defocusDistanceInMeters.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class DetectorImagePresenter(Observable, Observer):
    def __init__(self, imageSequence: ImageSequence) -> None:
        super().__init__()
        self._imageSequence = imageSequence

    @classmethod
    def createInstance(cls, imageSequence: ImageSequence) -> DetectorImagePresenter:
        presenter = cls(imageSequence)
        imageSequence.addObserver(presenter)
        return presenter

    def setCurrentDatasetIndex(self, index: int) -> None:
        self._imageSequence.setCurrentDatasetIndex(index)

    def getCurrentDatasetIndex(self) -> int:
        return self._imageSequence.getCurrentDatasetIndex()

    def getNumberOfImages(self) -> int:
        return len(self._imageSequence)

    def getImage(self, index: int) -> numpy.ndarray:
        try:
            return self._imageSequence[index]
        except IndexError:
            return None

    def update(self, observable: Observable) -> None:
        if observable is self._imageSequence:
            self.notifyObservers()
