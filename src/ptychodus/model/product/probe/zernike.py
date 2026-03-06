from __future__ import annotations
import logging

import numpy

from ptychodus.api.geometry import ZernikeMonomial
from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import generate_zernike_probe, rescale_probe_intensity

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ZernikeProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(settings, 'zernike')
        self._rng = rng
        self._settings = settings
        self._polynomial: list[ZernikeMonomial] = list()
        self._order = 0

        self.diameter_m = settings.disk_diameter_m.copy()
        self._add_parameter('diameter_m', self.diameter_m)

        self.set_order(1)

    def copy(self) -> ZernikeProbeBuilder:
        builder = ZernikeProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        builder._polynomial = self._polynomial.copy()
        builder._order = self._order
        return builder

    def set_order(self, order: int) -> None:
        if order < 1:
            logger.warning('Order must be strictly positive!')
            return

        if self._order == order:
            return

        self._polynomial.clear()

        for radial_degree in range(order):
            for angular_frequency in range(-radial_degree, 1 + radial_degree, 2):
                monomial = ZernikeMonomial(1 + 0j, radial_degree, angular_frequency)
                self._polynomial.append(monomial)

        self._order = order
        self.notify_observers()

    def get_order(self) -> int:
        return self._order

    def set_coefficient(self, idx: int, value: complex) -> None:
        monomial = self._polynomial[idx]
        self._polynomial[idx] = ZernikeMonomial(
            value, monomial.radial_degree, monomial.angular_frequency
        )

    def get_monomial(self, idx: int) -> ZernikeMonomial:
        return self._polynomial[idx]

    def __len__(self) -> int:
        return len(self._polynomial)

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe = rescale_probe_intensity(
            generate_zernike_probe(
                geometry_provider.get_probe_geometry(),
                self._polynomial,
                radius_m=self.diameter_m.get_value() / 2.0,
            ),
            geometry_provider.probe_photon_count,
        )
        return self._build_probe_modes(self._rng, probe, geometry_provider.num_scan_points)
