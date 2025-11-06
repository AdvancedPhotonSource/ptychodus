from __future__ import annotations
from enum import IntEnum

import numpy

from ptychodus.api.probe_positions import ProbePositionSequence, ProbePosition

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
    def is_triangular(self) -> bool:
        return self.value & 2 != 0

    @property
    def is_equilateral(self) -> bool:
        return self.value & 4 != 0


class CartesianProbePositionsBuilder(ProbePositionsBuilder):
    def __init__(
        self, variant: CartesianProbePositionsVariant, settings: ProbePositionsSettings
    ) -> None:
        super().__init__(settings, variant.name.lower())
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
        builder = CartesianProbePositionsBuilder(self._variant, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def is_equilateral(self) -> bool:
        return self._variant.is_equilateral

    def build(self) -> ProbePositionSequence:
        nx = self.num_points_x.get_value()
        ny = self.num_points_y.get_value()
        dx = self.step_size_x_m.get_value()

        if self._variant.is_equilateral:
            dy = dx

            if self._variant.is_triangular:
                dy *= numpy.sqrt(0.75)
        else:
            dy = self.step_size_y_m.get_value()

        point_list: list[ProbePosition] = list()

        for index in range(nx * ny):
            y, x = divmod(index, nx)

            if self._variant.is_snaked:
                if y & 1:
                    x = nx - 1 - x

            cx = (nx - 1) / 2
            cy = (ny - 1) / 2

            xf = (x - cx) * dx
            yf = (y - cy) * dy

            if self._variant.is_triangular:
                if y & 1:
                    xf += dx / 4
                else:
                    xf -= dx / 4

            point = ProbePosition(
                index=index,
                coordinate_x_m=xf,
                coordinate_y_m=yf,
            )
            point_list.append(point)

        return ProbePositionSequence(point_list)
