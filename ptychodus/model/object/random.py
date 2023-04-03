from decimal import Decimal
from typing import Any, Final

import numpy

from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from .repository import ObjectRepositoryItem
from .settings import ObjectSettings
from .sizer import ObjectSizer


class RandomObjectRepositoryItem(ObjectRepositoryItem):
    NAME: Final[str] = 'Random'

    def __init__(self, rng: numpy.random.Generator, sizer: ObjectSizer) -> None:
        super().__init__()
        self._rng = rng
        self._sizer = sizer
        self._extraPaddingX = 0
        self._extraPaddingY = 0
        self._amplitudeMean = Decimal(1) / 2
        self._amplitudeStandardDeviation = Decimal()
        self._randomizePhase = False

    @property
    def nameHint(self) -> str:
        return self.NAME

    @property
    def initializer(self) -> str:
        return self.NAME

    @property
    def canSelect(self) -> bool:
        return True

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        self._extraPaddingX = settings.extraPaddingX.value
        self._extraPaddingY = settings.extraPaddingY.value
        self._amplitudeMean = settings.amplitudeMean.value
        self._amplitudeStandardDeviation = settings.amplitudeStandardDeviation.value
        self._randomizePhase = settings.randomizePhase.value
        self.notifyObservers()

    def syncToSettings(self, settings: ObjectSettings) -> None:
        settings.extraPaddingX.value = self._extraPaddingX
        settings.extraPaddingY.value = self._extraPaddingY
        settings.amplitudeMean.value = self._amplitudeMean
        settings.amplitudeStandardDeviation.value = self._amplitudeStandardDeviation
        settings.randomizePhase.value = self._randomizePhase

    @property
    def _dtype(self) -> numpy.dtype[Any]:
        return numpy.dtype(complex)

    def getDataType(self) -> str:
        return str(self._dtype)

    def getExtentInPixels(self) -> ImageExtent:
        return self._sizer.getObjectExtent()

    def getSizeInBytes(self) -> int:
        return self._dtype.itemsize * self._sizer.getObjectExtent().size

    def getArray(self) -> ObjectArrayType:
        # FIXME cache this
        extraPaddingExtent = ImageExtent(self._extraPaddingX, self._extraPaddingY)
        paddedObjectExtent = self._sizer.getObjectExtent() + extraPaddingExtent

        size = paddedObjectExtent.shape
        amplitude = self._rng.normal(float(self._amplitudeMean),
                                     float(self._amplitudeStandardDeviation), size)
        phase = self._rng.uniform(0, 2 * numpy.pi, size=size) \
                if self._randomizePhase else numpy.zeros_like(amplitude)
        array: ObjectArrayType = amplitude * numpy.exp(1j * phase)
        return array

    def getExtraPaddingX(self) -> int:
        return self._extraPaddingX

    def setExtraPaddingX(self, value: int) -> None:
        if self._extraPaddingX != value:
            self._extraPaddingX = value
            self.notifyObservers()

    def getExtraPaddingY(self) -> int:
        return self._extraPaddingY

    def setExtraPaddingY(self, value: int) -> None:
        if self._extraPaddingY != value:
            self._extraPaddingY = value
            self.notifyObservers()

    def getAmplitudeMean(self) -> Decimal:
        return self._amplitudeMean

    def setAmplitudeMean(self, mean: Decimal) -> None:
        if self._amplitudeMean != mean:
            self._amplitudeMean = mean
            self.notifyObservers()

    def getAmplitudeStandardDeviation(self) -> Decimal:
        return self._amplitudeStandardDeviation

    def setAmplitudeStandardDeviation(self, stddev: Decimal) -> None:
        if self._amplitudeStandardDeviation != stddev:
            self._amplitudeStandardDeviation = stddev
            self.notifyObservers()

    def isRandomizePhaseEnabled(self) -> bool:
        return self._randomizePhase

    def setRandomizePhaseEnabled(self, enabled: bool) -> None:
        if self._randomizePhase != enabled:
            self._randomizePhase = enabled
            self.notifyObservers()
