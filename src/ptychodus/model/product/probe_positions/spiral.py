from __future__ import annotations

import numpy

from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.probe_positions_gen import generate_spiral_probe_positions

from .builder import ProbePositionsBuilder
from .settings import ProbePositionsSettings


class SpiralProbePositionsBuilder(ProbePositionsBuilder):
    """https://doi.org/10.1364/OE.22.012634"""

    def __init__(self, rng: numpy.random.Generator, settings: ProbePositionsSettings) -> None:
        super().__init__(rng, settings, 'spiral')
        self._rng = rng
        self._settings = settings

        self.num_points = settings.num_points_x.copy()
        self.num_points.set_value(
            settings.num_points_x.get_value() * settings.num_points_y.get_value()
        )
        self._add_parameter('num_points', self.num_points)

        self._num_points = settings.num_points_y.copy()
        self._num_points.set_value(1)
        self._add_parameter('_num_points', self._num_points)

        self.radius_scalar_m = settings.radius_scalar_m.copy()
        self._add_parameter('radius_scalar_m', self.radius_scalar_m)

    def copy(self) -> SpiralProbePositionsBuilder:
        builder = SpiralProbePositionsBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> ProbePositionSequence:
        positions = generate_spiral_probe_positions(
            self.num_points.get_value(), self.radius_scalar_m.get_value()
        )
        return self._create_position_sequence(positions)
