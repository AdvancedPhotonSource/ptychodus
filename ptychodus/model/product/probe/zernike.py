from __future__ import annotations
from dataclasses import dataclass
import logging

import numpy
import numpy.typing
import scipy.special

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.typing import RealArrayType

from .builder import ProbeBuilder
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


class ZernikeProbeBuilder(ProbeBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'zernike')
        self._settings = settings
        self._polynomials: list[ZernikePolynomial] = list()
        self._order = 0

        self.diameterInMeters = settings.diskDiameterInMeters.copy()
        self._addParameter('diameter_m', self.diameterInMeters)

        # TODO init zernike coefficients from settings
        self.coefficients = self.createComplexArrayParameter('coefficients', [1 + 0j])

        self.setOrder(1)

    def copy(self) -> ZernikeProbeBuilder:
        builder = ZernikeProbeBuilder(self._settings)
        builder.diameterInMeters.setValue(self.diameterInMeters.getValue())
        builder.coefficients.setValue(self.coefficients.getValue())
        builder.setOrder(self.getOrder())
        return builder

    def setOrder(self, order: int) -> None:
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
            coef = list(self.coefficients.getValue())
            coef += [0j] * (npoly - ncoef)
            self.coefficients.setValue(coef)

        self._order = order
        self.notifyObservers()

    def getOrder(self) -> int:
        return self._order

    def setCoefficient(self, idx: int, value: complex) -> None:
        self.coefficients[idx] = value

    def getCoefficient(self, idx: int) -> complex:
        return self.coefficients[idx]

    def getPolynomial(self, idx: int) -> ZernikePolynomial:
        return self._polynomials[idx]

    def __len__(self) -> int:
        return min(len(self.coefficients), len(self._polynomials))

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        radius = self.diameterInMeters.getValue() / 2.0
        distance = numpy.hypot(coords.positionYInMeters, coords.positionXInMeters) / radius
        angle = numpy.arctan2(coords.positionYInMeters, coords.positionXInMeters)
        array = numpy.zeros_like(distance, dtype=complex)

        for coef, poly in zip(self.coefficients, self._polynomials):
            array += numpy.multiply(coef, poly(distance, angle))

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
