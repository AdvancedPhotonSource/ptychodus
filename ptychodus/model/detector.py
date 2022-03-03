from __future__ import annotations
from pathlib import Path
from decimal import Decimal

import numpy

from .image import ImageSequence, CropSettings
from .observer import Observable
from .settings import SettingsRegistry, SettingsGroup
from .velociprobe import *


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
    def __init__(self, detectorSettings: DetectorSettings, cropSettings: CropSettings) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._cropSettings = cropSettings

    @classmethod
    def createInstance(cls, detectorSettings: DetectorSettings,
                       cropSettings: CropSettings) -> Detector:
        detector = cls(detectorSettings, cropSettings)
        detectorSettings.detectorDistanceInMeters.addObserver(detector)
        detectorSettings.pixelSizeXInMeters.addObserver(detector)
        detectorSettings.pixelSizeYInMeters.addObserver(detector)
        cropSettings.extentXInPixels.addObserver(detector)
        cropSettings.extentYInPixels.addObserver(detector)
        return detector

    @property
    def distanceToObjectInMeters(self) -> Decimal:
        return self._detectorSettings.detectorDistanceInMeters.value

    @property
    def extentXInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeXInMeters.value \
                * self._cropSettings.extentXInPixels.value

    @property
    def extentYInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeYInMeters.value \
                * self._cropSettings.extentYInPixels.value

    def update(self, observable: Observable) -> None:
        if observable is self._detectorSettings.detectorDistanceInMeters:
            self.notifyObservers()
        elif observable is self._detectorSettings.pixelSizeXInMeters:
            self.notifyObservers()
        elif observable is self._detectorSettings.pixelSizeYInMeters:
            self.notifyObservers()
        elif observable is self._cropSettings.extentXInPixels:
            self.notifyObservers()
        elif observable is self._cropSettings.extentYInPixels:
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


class DatasetPresenter(Observable, Observer):
    def __init__(self, settings: DetectorSettings, velociprobeReader: VelociprobeReader) -> None:
        super().__init__()
        self._settings = settings
        self._velociprobeReader = velociprobeReader

    @classmethod
    def createInstance(cls, settings: DetectorSettings,
                       velociprobeReader: VelociprobeReader) -> DatasetPresenter:
        presenter = cls(settings, velociprobeReader)
        velociprobeReader.addObserver(presenter)
        return presenter

    def getDatasetName(self, index: int) -> str:
        datafile = self._velociprobeReader.entryGroup.data[index]
        return datafile.name

    def getDatasetState(self, index: int) -> DatasetState:
        datafile = self._velociprobeReader.entryGroup.data[index]
        return datafile.getState()

    def getNumberOfDatasets(self) -> int:
        return 0 if self._velociprobeReader.entryGroup is None \
                else len(self._velociprobeReader.entryGroup.data)

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self._settings.dataPath.value = self._velociprobeReader.masterFilePath
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
