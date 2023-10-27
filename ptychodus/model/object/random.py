from decimal import Decimal
from typing import Final

import numpy

from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.object import Object
from .repository import ObjectInitializer
from .settings import ObjectSettings
from .sizer import ObjectSizer


class RandomObjectInitializer(ObjectInitializer):
    SIMPLE_NAME: Final[str] = 'Random'
    DISPLAY_NAME: Final[str] = 'Random'
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, rng: numpy.random.Generator, sizer: ObjectSizer) -> None:
        super().__init__()
        self._rng = rng
        self._sizer = sizer
        self._extraPaddingX = 1
        self._extraPaddingY = 1
        self._amplitudeMean = Decimal(1) / 2
        self._amplitudeDeviation = Decimal()
        self._phaseDeviation = Decimal()
        self._numberOfLayers = 1
        self._layerDistanceInMeters = Decimal()

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        self._extraPaddingX = settings.extraPaddingX.value
        self._extraPaddingY = settings.extraPaddingY.value
        self._amplitudeMean = settings.amplitudeMean.value
        self._amplitudeDeviation = settings.amplitudeDeviation.value
        self._phaseDeviation = settings.phaseDeviation.value
        self._numberOfLayers = settings.numberOfLayers.value
        self._layerDistanceInMeters = settings.layerDistanceInMeters.value
        self.notifyObservers()

    def syncToSettings(self, settings: ObjectSettings) -> None:
        settings.extraPaddingX.value = self._extraPaddingX
        settings.extraPaddingY.value = self._extraPaddingY
        settings.amplitudeMean.value = self._amplitudeMean
        settings.amplitudeDeviation.value = self._amplitudeDeviation
        settings.phaseDeviation.value = self._phaseDeviation
        settings.numberOfLayers.value = self._numberOfLayers
        settings.layerDistanceInMeters.value = self._layerDistanceInMeters

    def __call__(self) -> Object:
        extraPaddingExtent = ImageExtent(2 * self._extraPaddingX, 2 * self._extraPaddingY)
        paddedObjectExtent = self._sizer.getObjectExtentInPixels() + extraPaddingExtent
        objectShape = self.getNumberOfLayers(), *paddedObjectExtent.shape

        amplitude = self._rng.normal(float(self._amplitudeMean), float(self._amplitudeDeviation),
                                     objectShape)
        phase = self._rng.normal(0., float(self._phaseDeviation), objectShape)

        array = numpy.clip(amplitude, 0., 1.) * numpy.exp(1j * phase)
        distanceInMeters = float(self.getLayerDistanceInMeters())
        object_ = Object(array)

        for layer in range(object_.getNumberOfLayers()):
            object_.setLayerDistanceInMeters(layer, distanceInMeters)

        return object_

    def getExtraPaddingXLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getExtraPaddingX(self) -> int:
        limits = self.getExtraPaddingXLimits()
        return limits.clamp(self._extraPaddingX)

    def setExtraPaddingX(self, amount: int) -> None:
        if self._extraPaddingX != amount:
            self._extraPaddingX = amount
            self.notifyObservers()

    def getExtraPaddingYLimits(self) -> Interval[int]:
        return Interval[int](0, self.MAX_INT)

    def getExtraPaddingY(self) -> int:
        limits = self.getExtraPaddingYLimits()
        return limits.clamp(self._extraPaddingY)

    def setExtraPaddingY(self, amount: int) -> None:
        if self._extraPaddingY != amount:
            self._extraPaddingY = amount
            self.notifyObservers()

    def getAmplitudeMeanLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getAmplitudeMean(self) -> Decimal:
        limits = self.getAmplitudeMeanLimits()
        return limits.clamp(self._amplitudeMean)

    def setAmplitudeMean(self, mean: Decimal) -> None:
        if self._amplitudeMean != mean:
            self._amplitudeMean = mean
            self.notifyObservers()

    def getAmplitudeDeviationLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getAmplitudeDeviation(self) -> Decimal:
        limits = self.getAmplitudeDeviationLimits()
        return limits.clamp(self._amplitudeDeviation)

    def setAmplitudeDeviation(self, stddev: Decimal) -> None:
        if self._amplitudeDeviation != stddev:
            self._amplitudeDeviation = stddev
            self.notifyObservers()

    def getPhaseDeviationLimits(self) -> Interval[Decimal]:
        pi = Decimal.from_float(numpy.pi)
        return Interval[Decimal](Decimal(0), pi)

    def getPhaseDeviation(self) -> Decimal:
        limits = self.getPhaseDeviationLimits()
        return limits.clamp(self._phaseDeviation)

    def setPhaseDeviation(self, stddev: Decimal) -> None:
        if self._phaseDeviation != stddev:
            self._phaseDeviation = stddev
            self.notifyObservers()

    def getNumberOfLayersLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfLayers(self) -> int:
        limits = self.getNumberOfLayersLimits()
        return limits.clamp(self._numberOfLayers)

    def setNumberOfLayers(self, number: int) -> None:
        if self._numberOfLayers != number:
            self._numberOfLayers = number
            self.notifyObservers()

    def getLayerDistanceLimitsInMeters(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getLayerDistanceInMeters(self) -> Decimal:
        limits = self.getLayerDistanceLimitsInMeters()
        return limits.clamp(self._layerDistanceInMeters)

    def setLayerDistanceInMeters(self, number: Decimal) -> None:
        if self._layerDistanceInMeters != number:
            self._layerDistanceInMeters = number
            self.notifyObservers()
