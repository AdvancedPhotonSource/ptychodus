"""NumPy array type aliases used throughout the API."""

from typing import Any, TypeAlias

import numpy
import numpy.typing

IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]
InexactArrayType: TypeAlias = numpy.typing.NDArray[numpy.inexact[Any]]
NumberArrayType: TypeAlias = numpy.typing.NDArray[numpy.number]
