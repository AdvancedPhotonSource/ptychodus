from typing import Any, Final, TypeAlias, overload

import numpy
import numpy.typing

# Type Aliases
IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]
InexactArrayType: TypeAlias = numpy.typing.NDArray[numpy.inexact[Any]]
NumberArrayType: TypeAlias = numpy.typing.NDArray[numpy.number]

# Mathematical Constants
BYTES_PER_MEGABYTE: Final[int] = 1000 * 1000
TWO_PI: Final[float] = 2.0 * numpy.pi
TWO_PI_J: Final[complex] = 2.0j * numpy.pi

# Physical Constants
# Source: https://physics.nist.gov/cuu/Constants/index.html
ELECTRON_VOLT_J: Final[float] = 1.602176634e-19
LIGHT_SPEED_M_PER_S: Final[float] = 299792458
PLANCK_CONSTANT_J_PER_HZ: Final[float] = 6.62607015e-34


@overload
def lerp(lower: float, upper: float, frac: float) -> float: ...
@overload
def lerp(lower: complex, upper: complex, frac: float) -> complex: ...
@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: float) -> RealArrayType: ...
@overload
def lerp(lower: RealArrayType, upper: RealArrayType, frac: RealArrayType) -> RealArrayType: ...
@overload
def lerp(lower: float, upper: float, frac: RealArrayType) -> RealArrayType: ...


def lerp(
    lower: InexactArrayType | complex,
    upper: InexactArrayType | complex,
    frac: RealArrayType | float,
) -> InexactArrayType | complex:
    return (1.0 - frac) * lower + frac * upper
