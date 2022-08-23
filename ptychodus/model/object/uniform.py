import numpy

from ...api.object import ObjectArrayType
from .sizer import ObjectSizer


class UniformObjectInitializer:

    def __init__(self, sizer: ObjectSizer) -> None:
        self._sizer = sizer

    def __call__(self) -> ObjectArrayType:
        size = self._sizer.getObjectExtent().shape
        return numpy.ones(size, dtype=complex)
