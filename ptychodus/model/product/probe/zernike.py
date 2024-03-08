from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypeAlias
import logging

import numpy
import numpy.typing
import scipy.special

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from .builder import ProbeBuilder

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]
ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

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
        return numpy.sin(-self.angular_frequency * angle) if self.angular_frequency < 0 \
                else numpy.cos(self.angular_frequency * angle)

    def __call__(self,
                 distance: RealArrayType,
                 angle: RealArrayType,
                 undefined_value: float = 0.) -> RealArrayType:
        rvalue = self._radial_polynomial(distance)
        avalue = self._angular_function(angle)
        nvalue_sq = self.radial_degree + 1

        if self.angular_frequency != 0:
            nvalue_sq *= 2

        return numpy.where(numpy.logical_and(0 < distance, distance <= 1),
                           numpy.sqrt(nvalue_sq) * rvalue * avalue, undefined_value)

    def __str__(self) -> str:
        return f'$Z_{self.radial_degree}^{{{self.angular_frequency:+d}}}$'


class ZernikeProbeBuilder(ProbeBuilder):

    def __init__(self, geometryProvider: ProbeGeometryProvider) -> None:
        super().__init__('Zernike')
        self._geometryProvider = geometryProvider
        self._coefficients: list[complex] = [1. + 0j]  # FIXME make parameter
        self._polynomials: list[ZernikePolynomial] = []

        self.diameterInMeters = self._registerRealParameter(
            'DiameterInMeters',
            1.e-6,
            minimum=0.,
        )

        self.setOrder(1)

    def copy(self, geometryProvider: ProbeGeometryProvider) -> ZernikeProbeBuilder:
        builder = ZernikeProbeBuilder(geometryProvider)
        # FIXME coefficients
        builder.diameterInMeters.setValue(self.diameterInMeters.getValue())
        return builder

    def setOrder(self, order: int) -> None:
        if order < 1:
            logger.warning('Order must be strictly positive!')
            return

        self._polynomials.clear()

        for radial_degree in range(order):
            for angular_frequency in range(-radial_degree, 1 + radial_degree, 2):
                poly = ZernikePolynomial(radial_degree, angular_frequency)
                logger.debug(str(poly))
                self._polynomials.append(poly)

        npoly = len(self._polynomials)
        ncoef = len(self._coefficients)

        if ncoef < npoly:
            self._coefficients += [0j] * (npoly - ncoef)

    def getOrder(self) -> int:
        npoly = len(self._polynomials)
        order = 0

        while True:
            order += 1

            if (order * (order + 1)) // 2 > npoly:
                break

        return order

    def getCoefficient(self, idx: int) -> complex:
        return self._coefficients[idx]

    def getPolynomial(self, idx: int) -> ZernikePolynomial:
        return self._polynomials[idx]

    def build(self) -> Probe:
        geometry = self._geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        radius = self.diameterInMeters.getValue() / 2.
        distance = numpy.hypot(coords.positionYInMeters, coords.positionXInMeters) / radius
        angle = numpy.arctan2(coords.positionYInMeters, coords.positionXInMeters)

        array = numpy.zeros_like(distance, dtype=complex)

        for coef, poly in zip(self._coefficients, self._polynomials):
            array += numpy.multiply(coef, poly(distance, angle))

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
