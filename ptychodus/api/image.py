from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable

import numpy
import numpy.typing

ComplexArrayType = numpy.typing.NDArray[numpy.complexfloating]
RealArrayType = numpy.typing.NDArray[numpy.floating]


class ScalarTransformation(Callable[[RealArrayType], RealArrayType]):

    @abstractproperty
    def name(self) -> str:
        pass


class ComplexToRealStrategy(Callable[[ComplexArrayType, ScalarTransformation], RealArrayType]):

    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def isCyclic(self) -> bool:
        pass

    @abstractproperty
    def isColorized(self) -> bool:
        pass
