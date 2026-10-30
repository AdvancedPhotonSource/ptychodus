"""Generic, domain-agnostic NumPy array type aliases.

Aliases here name an *element-dtype family* (boolean / integer / floating /
complex / inexact / number) over an unspecified shape and are safe to use
anywhere a function takes or returns "some real array" without a further contract.

Shape- or domain-specific aliases do **not** belong here. When an array has
a fixed rank and a specific meaning tied to a domain concept — e.g. a single
diffraction pattern, a stack of them, or a bad-pixel mask — declare the
alias next to the class that produces or consumes it. See ``BadPixels``,
``DiffractionPattern``, ``DiffractionPatterns``, and ``DiffractionIndexes``
in :mod:`ptychodus.api.diffraction` for the reference pattern.
"""

from typing import TypeAlias

import numpy
import numpy.typing

BooleanArrayType: TypeAlias = numpy.typing.NDArray[numpy.bool]
IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating]
InexactArrayType: TypeAlias = numpy.typing.NDArray[numpy.inexact]
NumberArrayType: TypeAlias = numpy.typing.NDArray[numpy.number]

AnyArrayType: TypeAlias = numpy.typing.NDArray[numpy.generic]
"""An array of any dtype, for code that inspects one without constraining it.

Reach for this only where no dtype family is promised: a step that branches on
``dtype`` to decide what it was handed, or a file-format boundary that returns
whatever a dataset happens to hold. It is a statement that the dtype is unknown,
not a way to skip naming one that is known.
"""
