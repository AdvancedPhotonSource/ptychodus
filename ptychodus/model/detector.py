from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from typing import overload, Union
import logging

import numpy

from ..api.data import DataArrayType, DatasetState, DiffractionDataset
from ..api.geometry import Interval
from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup
from .data import ActiveDataFile, DetectorSettings, NullDiffractionDataset

logger = logging.getLogger(__name__)


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


class DiffractionDatasetPresenter(Observable, Observer):

    def __init__(self, dataFile: ActiveDataFile) -> None:
        super().__init__()
        self._dataFile = dataFile
        self._dataset: DiffractionDataset = NullDiffractionDataset()
        self._datasetIndex = 0

    @classmethod
    def createInstance(cls, dataFile: ActiveDataFile) -> DiffractionDatasetPresenter:
        presenter = cls(dataFile)
        dataFile.addObserver(presenter)
        return presenter

    def setCurrentDatasetIndex(self, index: int) -> None:
        try:
            dataset = self._dataFile[index]
        except IndexError:
            logger.exception('Invalid Dataset Index')
            return

        self._dataset.removeObserver(self)
        self._dataset = dataset
        self._dataset.addObserver(self)
        self._datasetIndex = index

        self.notifyObservers()

    def getCurrentDatasetIndex(self) -> int:
        return self._datasetIndex

    def getNumberOfImages(self) -> int:
        return len(self._dataset)

    def getImage(self, index: int) -> DataArrayType:
        return self._dataset[index]

    def update(self, observable: Observable) -> None:
        if observable is self._dataFile:
            self._dataset.removeObserver(self)
            self._dataset = NullDiffractionDataset()
            self._datasetIndex = 0
            self.notifyObservers()
        elif observable is self._dataset:
            self.notifyObservers()
