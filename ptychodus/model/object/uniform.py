from __future__ import annotations

import numpy

from ...api.object import ObjectArrayType
from .initializer import ObjectInitializer
from .settings import ObjectSettings
from .sizer import ObjectSizer


class UniformObjectInitializer(ObjectInitializer):

    def __init__(self, sizer: ObjectSizer) -> None:
        self._sizer = sizer

    @classmethod
    def createInstance(cls, settings: ObjectSettings,
                       sizer: ObjectSizer) -> UniformObjectInitializer:
        initializer = cls(sizer)
        initializer.syncFromSettings(settings)
        return initializer

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        super().syncFromSettings(settings)

    def syncToSettings(self, settings: ObjectSettings) -> None:
        super().syncToSettings(settings)

    @property
    def displayName(self) -> str:
        return 'Uniform'

    @property
    def simpleName(self) -> str:
        return super().simpleName

    def __call__(self) -> ObjectArrayType:
        shape = self._sizer.getObjectExtent().shape
        return numpy.full(shape, 0.5, dtype=complex)
