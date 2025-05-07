from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy
import numpy.typing
import scipy.special

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.typing import RealArrayType

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ZernikePolynomial:
    radial_degree: int  # n
    angular_frequency: int  # m

    @property
    def spatial_frequencey(self) -> int:
        return self.radial_degree + abs(self.angular_frequency)

    def _radial_polynomial(self, distance: RealArrayType) -> RealArrayType:
        n_minus_m = self.radial_degree - abs(self.angular_frequency)
        half_n_minus_m = n_minus_m // 2
        sgn = 1

        values = numpy.zeros_like(distance)

        for k in range(half_n_minus_m + 1):
            n_minus_k = self.radial_degree - k
            n_minus_2k = self.radial_degree - 2 * k

            coef = sgn
            coef *= scipy.special.binom(n_minus_k, k)
            coef *= scipy.special.binom(n_minus_2k, half_n_minus_m - k)
            coef = int(coef)  # NOTE!

            values += numpy.multiply(coef, numpy.power(distance, n_minus_2k))

            sgn = -sgn

        return values

    def _angular_function(self, angle: RealArrayType) -> RealArrayType:
        return (
            numpy.sin(-self.angular_frequency * angle)
            if self.angular_frequency < 0
            else numpy.cos(self.angular_frequency * angle)
        )

    def __call__(
        self, distance: RealArrayType, angle: RealArrayType, undefined_value: float = 0.0
    ) -> RealArrayType:
        rvalue = self._radial_polynomial(distance)
        avalue = self._angular_function(angle)
        nvalue_sq = self.radial_degree + 1

        if self.angular_frequency != 0:
            nvalue_sq *= 2

        return numpy.where(
            numpy.logical_and(0 < distance, distance <= 1),
            numpy.sqrt(nvalue_sq) * rvalue * avalue,
            undefined_value,
        )

    def __str__(self) -> str:
        return f'$Z_{{{self.radial_degree}}}^{{{self.angular_frequency:+d}}}$'


class ZernikeProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'zernike')
        self._settings = settings
        self._polynomials: list[ZernikePolynomial] = list()
        self._order = 0

        self.diameter_m = settings.disk_diameter_m.copy()
        self._add_parameter('diameter_m', self.diameter_m)

        # TODO init zernike coefficients from settings
        self.coefficients = self.create_complex_sequence_parameter('coefficients', [1 + 0j])

        self.set_order(1)

    def copy(self) -> ZernikeProbeBuilder:
        builder = ZernikeProbeBuilder(self._settings)
        builder.diameter_m.set_value(self.diameter_m.get_value())
        builder.coefficients.set_value(self.coefficients.get_value())
        builder.set_order(self.get_order())
        return builder

    def set_order(self, order: int) -> None:
        if order < 1:
            logger.warning('Order must be strictly positive!')
            return

        if self._order == order:
            return

        self._polynomials.clear()

        for radial_degree in range(order):
            for angular_frequency in range(-radial_degree, 1 + radial_degree, 2):
                poly = ZernikePolynomial(radial_degree, angular_frequency)
                self._polynomials.append(poly)

        npoly = len(self._polynomials)
        ncoef = len(self.coefficients)

        if ncoef < npoly:
            coef = list(self.coefficients.get_value())
            coef += [0j] * (npoly - ncoef)
            self.coefficients.set_value(coef)

        self._order = order
        self.notify_observers()

    def get_order(self) -> int:
        return self._order

    def set_coefficient(self, idx: int, value: complex) -> None:
        self.coefficients[idx] = value

    def get_coefficient(self, idx: int) -> complex:
        return self.coefficients[idx]

    def get_polynomial(self, idx: int) -> ZernikePolynomial:
        return self._polynomials[idx]

    def __len__(self) -> int:
        return min(len(self.coefficients), len(self._polynomials))

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        geometry = geometry_provider.get_probe_geometry()
        coords = self.get_transverse_coordinates(geometry)

        radius = self.diameter_m.get_value() / 2.0
        distance = numpy.hypot(coords.position_y_m, coords.position_x_m) / radius
        angle = numpy.arctan2(coords.position_y_m, coords.position_x_m)
        array = numpy.zeros_like(distance, dtype=complex)

        for coef, poly in zip(self.coefficients, self._polynomials):
            array += numpy.multiply(coef, poly(distance, angle))

        return ProbeSequence(
            array=self.normalize(array),
            opr_weights=None,
            pixel_geometry=geometry.get_pixel_geometry(),
        )
