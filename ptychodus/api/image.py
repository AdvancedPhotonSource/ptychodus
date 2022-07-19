from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable

import numpy
import numpy.typing

RealArrayType = numpy.typing.NDArray[numpy.floating]


class ScalarTransformation(Callable[[RealArrayType], RealArrayType]):

    @abstractproperty
    def name(self) -> str:
        pass

