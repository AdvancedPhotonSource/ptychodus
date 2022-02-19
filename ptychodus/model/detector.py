from __future__ import annotations
from pathlib import Path
from decimal import Decimal

import numpy

from .image import ImageSequence
from .observer import Observable
from .settings import SettingsRegistry, SettingsGroup
from .velociprobe import *


class DetectorSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
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
        return self._detectorSettings.pixelSizeXInMeters.value * self._cropSettings.extentXInPixels.value

    @property
    def extentYInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeYInMeters.value * self._cropSettings.extentYInPixels.value

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


class DetectorParametersPresenter(Observer, Observable):
    def __init__(self, detectorSettings: DetectorSettings, cropSettings: CropSettings) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._cropSettings = cropSettings

    @classmethod
    def createInstance(cls, detectorSettings: DetectorSettings,
                       cropSettings: CropSettings) -> DetectorParametersPresenter:
        presenter = cls(detectorSettings, cropSettings)
        detectorSettings.addObserver(presenter)
        cropSettings.addObserver(presenter)
        return presenter

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeXInMeters.value

    def setPixelSizeXInMeters(self, value: Decimal) -> None:
        self._detectorSettings.pixelSizeXInMeters.value = value

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._detectorSettings.pixelSizeYInMeters.value

    def setPixelSizeYInMeters(self, value: Decimal) -> None:
        self._detectorSettings.pixelSizeYInMeters.value = value

    def getDetectorDistanceInMeters(self) -> Decimal:
        return self._detectorSettings.detectorDistanceInMeters.value

    def setDetectorDistanceInMeters(self, value: Decimal) -> None:
        self._detectorSettings.detectorDistanceInMeters.value = value

    def getDefocusDistanceInMeters(self) -> Decimal:
        return self._detectorSettings.defocusDistanceInMeters.value

    def setDefocusDistanceInMeters(self, value: Decimal) -> None:
        self._detectorSettings.defocusDistanceInMeters.value = value

    def isCropEnabled(self) -> bool:
        return self._cropSettings.cropEnabled.value

    def setCropEnabled(self, value: bool) -> None:
        self._cropSettings.cropEnabled.value = value

    def getCropCenterXInPixels(self) -> int:
        return self._cropSettings.centerXInPixels.value

    def setCropCenterXInPixels(self, value: int) -> None:
        self._cropSettings.centerXInPixels.value = value

    def getCropCenterYInPixels(self) -> int:
        return self._cropSettings.centerYInPixels.value

    def setCropCenterYInPixels(self, value: int) -> None:
        self._cropSettings.centerYInPixels.value = value

    def getCropExtentXInPixels(self) -> int:
        return self._cropSettings.extentXInPixels.value

    def setCropExtentXInPixels(self, value: int) -> None:
        self._cropSettings.extentXInPixels.value = value

    def getCropExtentYInPixels(self) -> int:
        return self._cropSettings.extentYInPixels.value

    def setCropExtentYInPixels(self, value: int) -> None:
        self._cropSettings.extentYInPixels.value = value

    def update(self, observable: Observable) -> None:
        if observable is self._detectorSettings:
            self.notifyObservers()
        elif observable is self._cropSettings:
            self.notifyObservers()


class DetectorDatasetPresenter(Observable, Observer):
    def __init__(self, velociprobeReader: VelociprobeReader) -> None:
        super().__init__()
        self._velociprobeReader = velociprobeReader

    @classmethod
    def createInstance(cls, velociprobeReader: VelociprobeReader) -> DetectorDatasetPresenter:
        presenter = cls(velociprobeReader)
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
            self.notifyObservers()


class VelociprobeImageSequence(ImageSequence):
    def __init__(self, velociprobeReader: VelociprobeReader) -> None:
        super().__init__()
        self._velociprobeReader = velociprobeReader
        self._datasetImageList = None
        self._datasetIndex = -1

    @classmethod
    def createInstance(cls, velociprobeReader: VelociprobeReader) -> VelociprobeImageSequence:
        imageSequence = cls(velociprobeReader)
        imageSequence._updateImages()
        velociprobeReader.addObserver(imageSequence)
        return imageSequence

    def setCurrentDatasetIndex(self, index: int) -> None:
        if index < 0:
            raise IndexError('Current dataset index must be non-negative.')

        self._datasetIndex = index
        self._updateImages()

    def getCurrentDatasetIndex(self) -> int:
        return self._datasetIndex

    def getWidth(self) -> int:
        value = 0

        if self._datasetImageList:
            value = self._datasetImageList[0].shape[1]

        return value

    def getHeight(self) -> int:
        value = 0

        if self._datasetImageList:
            value = self._datasetImageList[0].shape[0]

        return value

    def __getitem__(self, index: int) -> numpy.ndarray:
        return self._datasetImageList[index]

    def __len__(self) -> int:
        return len(self._datasetImageList)

    def _updateImages(self) -> None:
        self._datasetImageList = list()

        if self._velociprobeReader.entryGroup is None:
            self._datasetIndex = 0
            return
        elif self._datasetIndex >= len(self._velociprobeReader.entryGroup.data):
            self._datasetIndex = 0

        datafile = self._velociprobeReader.entryGroup.data[self._datasetIndex]

        with h5py.File(datafile.filePath, 'r') as h5File:
            item = h5File.get(datafile.dataPath)

            if isinstance(item, h5py.Dataset):
                data = item[()]

                for imslice in data:
                    image = numpy.copy(imslice)
                    self._datasetImageList.append(image)
            else:
                raise TypeError('Data path does not refer to a dataset.')

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self._updateImages()


class CroppedImageSequence(ImageSequence):
    def __init__(self, settings: CropSettings, imageSequence: ImageSequence) -> None:
        super().__init__()
        self._settings = settings
        self._imageSequence = imageSequence

    @classmethod
    def createInstance(cls, settings: CropSettings,
                       imageSequence: ImageSequence) -> CroppedImageSequence:
        croppedImageSequence = cls(settings, imageSequence)
        settings.addObserver(croppedImageSequence)
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

        if self._settings.cropEnabled.value == True:
            radiusX = self._settings.extentXInPixels.value // 2
            xmin = self._settings.centerXInPixels.value - radiusX
            xmax = self._settings.centerXInPixels.value + radiusX

            radiusY = self._settings.extentYInPixels.value // 2
            ymin = self._settings.centerYInPixels.value - radiusY
            ymax = self._settings.centerYInPixels.value + radiusY

            img = numpy.copy(img[ymin:ymax, xmin:xmax])

        return img

    def __len__(self) -> int:
        return len(self._imageSequence)

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._imageSequence:
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
