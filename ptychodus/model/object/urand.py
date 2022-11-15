from __future__ import annotations

import numpy

from ...api.object import ObjectArrayType
from .initializer import ObjectInitializer
from .settings import ObjectSettings
from .sizer import ObjectSizer


class UniformRandomObjectInitializer(ObjectInitializer):

    def __init__(self, rng: numpy.random.Generator, sizer: ObjectSizer) -> None:
        self._rng = rng
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: ObjectSettings, rng: numpy.random.Generator,
                       sizer: ObjectSizer) -> UniformRandomObjectInitializer:
        initializer = cls(rng, sizer)
        initializer.syncFromSettings(settings)
        return initializer

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        super().syncFromSettings(settings)

    def syncToSettings(self, settings: ObjectSettings) -> None:
        super().syncToSettings(settings)

    @property
    def displayName(self) -> str:
        return 'Random'

    @property
    def simpleName(self) -> str:
        return super().simpleName

    def __call__(self) -> ObjectArrayType:
        size = self._sizer.getObjectExtent().shape
        magnitude = numpy.sqrt(self._rng.uniform(low=0., high=1e-6, size=size))
        phase = self._rng.uniform(low=0., high=2. * numpy.pi, size=size)
        return magnitude * numpy.exp(1.j * phase)
