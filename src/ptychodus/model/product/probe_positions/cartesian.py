from __future__ import annotations
from enum import IntEnum

import numpy

from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.probe_positions_gen import generate_cartesian_probe_positions

from .builder import ProbePositionsBuilder
from .settings import ProbePositionsSettings


class CartesianProbePositionsVariant(IntEnum):
    RECTANGULAR_RASTER = 0x0
    RECTANGULAR_SNAKE = 0x1
    TRIANGULAR_RASTER = 0x2
    TRIANGULAR_SNAKE = 0x3
    SQUARE_RASTER = 0x4
    SQUARE_SNAKE = 0x5
    HEXAGONAL_RASTER = 0x6
    HEXAGONAL_SNAKE = 0x7

    @property
    def is_snaked(self) -> bool:
        return self.value & 1 != 0

    @property
    def is_staggered(self) -> bool:
        return self.value & 2 != 0

    @property
    def is_equilateral(self) -> bool:
        return self.value & 4 != 0


class CartesianProbePositionsBuilder(ProbePositionsBuilder):
    def __init__(
        self,
        variant: CartesianProbePositionsVariant,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
    ) -> None:
        super().__init__(rng, settings, variant.name.lower())
        self._rng = rng
        self._variant = variant
        self._settings = settings

        self.num_points_x = settings.num_points_x.copy()
        self._add_parameter('num_points_x', self.num_points_x)

        self.num_points_y = settings.num_points_y.copy()
        self._add_parameter('num_points_y', self.num_points_y)

        self.step_size_x_m = settings.step_size_x_m.copy()
        self._add_parameter('step_size_x_m', self.step_size_x_m)

        self.step_size_y_m = settings.step_size_y_m.copy()
        self._add_parameter('step_size_y_m', self.step_size_y_m)

    def copy(self) -> CartesianProbePositionsBuilder:
        builder = CartesianProbePositionsBuilder(self._variant, self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def is_equilateral(self) -> bool:
        return self._variant.is_equilateral

    def build(self) -> ProbePositionSequence:
        step_size_x_m = self.step_size_x_m.get_value()

        if self._variant.is_equilateral:
            step_size_y_m = step_size_x_m

            if self._variant.is_staggered:
                step_size_y_m *= numpy.sqrt(0.75)
        else:
            step_size_y_m = self.step_size_y_m.get_value()

        positions = generate_cartesian_probe_positions(
            self.num_points_x.get_value(),
            self.num_points_y.get_value(),
            step_size_x_m,
            step_size_y_m,
            snake=self._variant.is_snaked,
            stagger=self._variant.is_staggered,
        )
        return self._create_position_sequence(positions)
