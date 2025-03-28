from typing import Any, TypeAlias

from scipy.integrate import quad
from scipy.special import fresnel, j0
import numpy
import numpy.typing

ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]
RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]


class SquareAperture:
    def __init__(self, width_m: float, wavelength_m: float) -> None:
        self._width_m = width_m
        self._wavelength_m = wavelength_m

    def get_fresnel_number(self, z_m: float) -> float:
        upper = (self._width_m / 2) ** 2
        lower = self._wavelength_m * z_m
        return upper / lower

    def _integral1d(self, r_m: RealArrayType, z_m: float) -> ComplexArrayType:
        sqrt2NF = numpy.sqrt(2 * self.get_fresnel_number(z_m))  # noqa: N806
        xi = 2 * r_m / self._width_m
        Sm, Cm = fresnel(sqrt2NF * (1 - xi))  # noqa: N806
        Sp, Cp = fresnel(sqrt2NF * (1 + xi))  # noqa: N806
        return (Cm + Cp + 1j * (Sm + Sp)) / numpy.sqrt(2)

    def diffract(self, x_m: RealArrayType, y_m: RealArrayType, z_m: float) -> ComplexArrayType:
        """Fresnel diffraction; see Goodman p.100"""
        assert x_m.shape == y_m.shape
        Ix = self._integral1d(x_m, z_m)  # noqa: N806
        Iy = self._integral1d(y_m, z_m)  # noqa: N806
        return Ix * Iy * numpy.exp(2j * numpy.pi * z_m / self._wavelength_m) / 1j


class CircularAperture:
    def __init__(self, diameter_m: float, wavelength_m: float) -> None:
        self._diameter_m = diameter_m
        self._wavelength_m = wavelength_m

    def get_fresnel_number(self, z_m: float) -> float:
        upper = (self._diameter_m / 2) ** 2
        lower = self._wavelength_m * z_m
        return upper / lower

    def diffract(self, x_m: RealArrayType, y_m: RealArrayType, z_m: float) -> ComplexArrayType:
        """Fresnel diffraction; see Goodman p.102"""
        assert x_m.shape == y_m.shape

        twopi = 2 * numpy.pi
        sqrtLZ = numpy.sqrt(self._wavelength_m * z_m)  # noqa: N806
        sqrtNF = numpy.sqrt(self.get_fresnel_number(z_m))  # noqa: N806

        rhop = rho / sqrtLZ
        rp = numpy.hypot(x_m, y_m) / sqrtLZ

        arg = twopi * rhop * rp
        integrand = rhop * numpy.exp(1j * numpy.pi * rhop**2) * j0(arg)
        integral = integrate(integrand, 0, sqrtNF)
        return twopi * integral
