from __future__ import annotations
import logging

import numpy

from ptychodus.api.geometry import HermiteMode
from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.probe_gen import generate_hermite_probe, rescale_probe_intensity

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class HermiteProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(settings, 'hermite')
        self._rng = rng
        self._settings = settings
        self._polynomial: list[HermiteMode] = list()

        self.width_m = settings.rectangle_width_m.copy()
        self._add_parameter('width_m', self.width_m)

        self.height_m = settings.rectangle_height_m.copy()
        self._add_parameter('height_m', self.height_m)

        self.order_x_max = settings.hermite_order_x.copy()
        self._add_parameter('order_x_max', self.order_x_max)

        self.order_y_max = settings.hermite_order_y.copy()
        self._add_parameter('order_y_max', self.order_y_max)

        self._rebuild_polynomial()

    def copy(self) -> HermiteProbeBuilder:
        builder = HermiteProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        builder._polynomial = self._polynomial.copy()
        return builder

    def _rebuild_polynomial(self) -> None:
        self._polynomial.clear()

        for order_x in range(self.order_x_max.get_value()):
            for order_y in range(self.order_y_max.get_value()):
                self._polynomial.append(HermiteMode(1 + 0j, order_x, order_y))

    def set_order_x(self, order: int) -> None:
        if order < 1:
            logger.warning('Order must be strictly positive!')
            return

        if self.order_x_max.get_value() == order:
            return

        self.order_x_max.set_value(order)
        self._rebuild_polynomial()
        self.notify_observers()

    def set_order_y(self, order: int) -> None:
        if order < 1:
            logger.warning('Order must be strictly positive!')
            return

        if self.order_y_max.get_value() == order:
            return

        self.order_y_max.set_value(order)
        self._rebuild_polynomial()
        self.notify_observers()

    def get_order_x(self) -> int:
        return self.order_x_max.get_value()

    def get_order_y(self) -> int:
        return self.order_y_max.get_value()

    def set_coefficient(self, idx: int, value: complex) -> None:
        mode = self._polynomial[idx]
        self._polynomial[idx] = HermiteMode(value, mode.order_x, mode.order_y)

    def get_mode(self, idx: int) -> HermiteMode:
        return self._polynomial[idx]

    def __len__(self) -> int:
        return len(self._polynomial)

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe = rescale_probe_intensity(
            generate_hermite_probe(
                geometry_provider.get_probe_geometry(),
                self._polynomial,
                width_m=self.width_m.get_value(),
                height_m=self.height_m.get_value(),
            ),
            geometry_provider.probe_photon_count,
        )
        return self._build_probe_modes(self._rng, probe, geometry_provider.num_scan_points)
