import numpy

from ...api.object import ObjectArrayType
from .sizer import ObjectSizer


class UniformRandomObjectInitializer:

    def __init__(self, rng: numpy.random.Generator, sizer: ObjectSizer) -> None:
        self._rng = rng
        self._sizer = sizer

    def __call__(self) -> ObjectArrayType:
        size = self._sizer.getObjectExtent().shape
        magnitude = numpy.sqrt(self._rng.uniform(low=0., high=1e-6, size=size))
        phase = self._rng.uniform(low=0., high=2. * numpy.pi, size=size)
        return magnitude * numpy.exp(1.j * phase)
