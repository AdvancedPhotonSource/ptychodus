from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable

import numpy
import numpy.typing

ComplexArrayType = numpy.typing.NDArray[numpy.complexfloating]
RealArrayType = numpy.typing.NDArray[numpy.floating]


# FIXME ComplexToImageStrategy; pass in ScalarTransformation?
class ComplexToRealStrategy(Callable[[ComplexArrayType], RealArrayType]):
    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def isCyclic(self) -> bool:
        pass


class ScalarTransformation(Callable[[RealArrayType], RealArrayType]):
    @abstractproperty
    def name(self) -> str:
        pass
