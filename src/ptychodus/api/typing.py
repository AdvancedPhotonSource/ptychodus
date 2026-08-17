"""Generic, domain-agnostic NumPy array type aliases.

Aliases here name an *element-dtype family* (integer / floating / complex /
inexact / number) over an unspecified shape and are safe to use anywhere a
function takes or returns "some real array" without a further contract.

Shape- or domain-specific aliases do **not** belong here. When an array has
a fixed rank and a specific meaning tied to a domain concept — e.g. a single
diffraction pattern, a stack of them, or a bad-pixel mask — declare the
alias next to the class that produces or consumes it. See ``BadPixels``,
``DiffractionPattern``, ``DiffractionPatterns``, and ``DiffractionIndexes``
in :mod:`ptychodus.api.diffraction` for the reference pattern.
"""

from typing import Any, TypeAlias

import numpy
import numpy.typing

IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]
InexactArrayType: TypeAlias = numpy.typing.NDArray[numpy.inexact[Any]]
NumberArrayType: TypeAlias = numpy.typing.NDArray[numpy.number]
