from typing import Final

import numpy

from ...api.object import ObjectArrayType
from .repository import ObjectInitializer
from .settings import ObjectSettings


class CompareObjectInitializer(ObjectInitializer):
    SIMPLE_NAME: Final[str] = 'Compare'
    DISPLAY_NAME: Final[str] = 'Compare'

    def __init__(self) -> None:
        super().__init__()

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ObjectSettings) -> None:
        pass

    def syncToSettings(self, settings: ObjectSettings) -> None:
        pass

    def __call__(self) -> ObjectArrayType:
        return numpy.zeros((64, 64), dtype=complex)  # FIXME
