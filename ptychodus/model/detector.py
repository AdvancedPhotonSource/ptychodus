from __future__ import annotations
from decimal import Decimal
import logging

import numpy

from ..api.data import DiffractionDataset, DataArrayType
from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup
from .data import ActiveDataFile
from .geometry import Interval

logger = logging.getLogger(__name__)


class DetectorSettings(Observable, Observer):
    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.numberOfPixelsX = settingsGroup.createIntegerEntry('NumberOfPixelsX', 1024)
        self.numberOfPixelsY = settingsGroup.createIntegerEntry('NumberOfPixelsY', 1024)
        self.pixelSizeXInMeters = settingsGroup.createRealEntry('PixelSizeXInMeters', '75e-6')
        self.pixelSizeYInMeters = settingsGroup.createRealEntry('PixelSizeYInMeters', '75e-6')
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '2')

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


class Detector(Observable, Observer):
    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> Detector:
        detector = cls(settings)
        settings.detectorDistanceInMeters.addObserver(detector)
        settings.pixelSizeXInMeters.addObserver(detector)
        settings.pixelSizeYInMeters.addObserver(detector)
        return detector

    @property
    def numberOfPixelsX(self) -> int:
        return self._settings.numberOfPixelsX.value

    @property
    def pixelSizeXInMeters(self) -> Decimal:
        return self._settings.pixelSizeXInMeters.value

    @property
    def numberOfPixelsY(self) -> int:
        return self._settings.numberOfPixelsY.value

    @property
    def pixelSizeYInMeters(self) -> Decimal:
        return self._settings.pixelSizeYInMeters.value

    @property
    def distanceToObjectInMeters(self) -> Decimal:
        return self._settings.detectorDistanceInMeters.value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
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


class DetectorPresenter(Observer, Observable):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: DetectorSettings) -> DetectorPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def getMinNumberOfPixelsX(self) -> int:
        return 0

    def getMaxNumberOfPixelsX(self) -> int:
        return self.MAX_INT

    def getNumberOfPixelsX(self) -> int:
        return self._settings.numberOfPixelsX.value

    def setNumberOfPixelsX(self, value: int) -> None:
        self._settings.numberOfPixelsX.value = value

    def getMinNumberOfPixelsY(self) -> int:
        return 0

    def getMaxNumberOfPixelsY(self) -> int:
        return self.MAX_INT

    def getNumberOfPixelsY(self) -> int:
        return self._settings.numberOfPixelsY.value

    def setNumberOfPixelsY(self, value: int) -> None:
        self._settings.numberOfPixelsY.value = value

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

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class ActiveDiffractionDataset(DiffractionDataset):
    def __init__(self, dataFile: ActiveDataFile, cropSizer: CropSizer) -> None:
        super().__init__()
        self._dataFile = dataFile
        self._cropSizer = cropSizer
        self._datasetIndex = 0
        self._dataset: Optional[DiffractionDataset] = None

    @classmethod
    def createInstance(cls, dataFile: ActiveDataFile,
                       cropSizer: CropSizer) -> ActiveDiffractionDataset:
        dataset = cls(dataFile, cropSizer)
        dataFile.addObserver(dataset)
        cropSizer.addObserver(dataset)
        return dataset

    def setDatasetIndex(self, index: int) -> None:
        try:
            dataset = self._dataFile[index]
        except IndexError:
            logger.exception('Invalid Dataset Index')
            return

        if self._dataset is not None:
            self._dataset.removeObserver(self)

        self._dataset = dataset
        self._dataset.addObserver(self)
        self._datasetIndex = index

        self.notifyObservers()

    def getDatasetIndex(self) -> int:
        return self._datasetIndex

    @property
    def datasetName(self) -> str:
        return '' if self._dataset is None else self._dataset.datasetName

    @property
    def datasetState(self) -> DatasetState:
        return DatasetState.NOT_FOUND if self._dataset is None else self._dataset.datasetState

    def getArray(self) -> DataArrayType:
        return numpy.empty((0, 0, 0)) if self._dataset is None else self._dataset.getArray()

    def __getitem__(self, index: int) -> DataArrayType:
        data = numpy.empty((0, 0))

        if self._dataset is not None:
            try:
                data = self._dataset[index]
            except IndexError:
                pass
            else:
                if self._cropSizer.isCropEnabled():
                    sliceX = self._cropSizer.getSliceX()
                    sliceY = self._cropSizer.getSliceY()
                    data = data[sliceY, sliceX].copy()

        return data

    def __len__(self) -> int:
        return 0 if self._dataset is None else len(self._dataset)

    def update(self, observable: Observable) -> None:
        if observable is self._dataFile:
            self._datasetIndex = 0
            self._dataset = None
            self.notifyObservers()
        elif observable is self._cropSizer:
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


class DiffractionDatasetPresenter(Observable, Observer):
    def __init__(self, dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._dataset = dataset

    @classmethod
    def createInstance(cls, dataset: ActiveDiffractionDataset) -> DiffractionDatasetPresenter:
        presenter = cls(dataset)
        dataset.addObserver(presenter)
        return presenter

    def setCurrentDatasetIndex(self, index: int) -> None:
        self._dataset.setDatasetIndex(index)

    def getCurrentDatasetIndex(self) -> int:
        return self._dataset.getDatasetIndex()

    def getNumberOfImages(self) -> int:
        return len(self._dataset)

    def getImage(self, index: int) -> DataArrayType:
        return self._dataset[index]

    def update(self, observable: Observable) -> None:
        if observable is self._dataset:
            self.notifyObservers()
