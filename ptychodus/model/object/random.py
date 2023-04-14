from decimal import Decimal
from typing import Final

import numpy

from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.object import ObjectArrayType
from .repository import ObjectInitializer
from .settings import ObjectSettings
from .sizer import ObjectSizer


class RandomObjectInitializer(ObjectInitializer):
    NAME: Final[str] = 'Random'

    def __init__(self, rng: numpy.random.Generator, sizer: ObjectSizer) -> None:
        super().__init__()
        self._rng = rng
        self._sizer = sizer
        self._extraPaddingX = 0
        self._extraPaddingY = 0
        self._amplitudeMean = Decimal(1) / 2
        self._amplitudeDeviation = Decimal()
        self._randomizePhase = False

    @property
    def simpleName(self) -> str:
        return self.NAME

    @property
    def displayName(self) -> str:
        return self.NAME

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        self._extraPaddingX = settings.extraPaddingX.value
        self._extraPaddingY = settings.extraPaddingY.value
        self._amplitudeMean = settings.amplitudeMean.value
        self._amplitudeDeviation = settings.amplitudeDeviation.value
        self._randomizePhase = settings.randomizePhase.value
        self.notifyObservers()

    def syncToSettings(self, settings: ObjectSettings) -> None:
        settings.extraPaddingX.value = self._extraPaddingX
        settings.extraPaddingY.value = self._extraPaddingY
        settings.amplitudeMean.value = self._amplitudeMean
        settings.amplitudeDeviation.value = self._amplitudeDeviation
        settings.randomizePhase.value = self._randomizePhase

    def __call__(self) -> ObjectArrayType:
        extraPaddingExtent = ImageExtent(self._extraPaddingX, self._extraPaddingY)
        paddedObjectExtent = self._sizer.getObjectExtent() + extraPaddingExtent

        size = paddedObjectExtent.shape
        amplitude = numpy.clip(
            self._rng.normal(float(self._amplitudeMean), float(self._amplitudeDeviation), size),
            0., 1.)
        phase = self._rng.uniform(0, 2 * numpy.pi, size=size) \
                if self._randomizePhase else numpy.zeros_like(amplitude)
        return amplitude * numpy.exp(1j * phase)

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

    def getAmplitudeMeanLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getAmplitudeMean(self) -> Decimal:
        return self._amplitudeMean

    def setAmplitudeMean(self, mean: Decimal) -> None:
        if self._amplitudeMean != mean:
            self._amplitudeMean = mean
            self.notifyObservers()

    def getAmplitudeDeviationLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getAmplitudeDeviation(self) -> Decimal:
        return self._amplitudeDeviation

    def setAmplitudeDeviation(self, stddev: Decimal) -> None:
        if self._amplitudeDeviation != stddev:
            self._amplitudeDeviation = stddev
            self.notifyObservers()

    def isRandomizePhaseEnabled(self) -> bool:
        return self._randomizePhase

    def setRandomizePhaseEnabled(self, enabled: bool) -> None:
        if self._randomizePhase != enabled:
            self._randomizePhase = enabled
            self.notifyObservers()
