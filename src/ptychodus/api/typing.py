from typing import Any, TypeAlias

import numpy
import numpy.typing

IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

NumberTypes: TypeAlias = numpy.integer[Any] | numpy.floating[Any] | numpy.complexfloating[Any, Any]
NumberArrayType: TypeAlias = numpy.typing.NDArray[NumberTypes]
