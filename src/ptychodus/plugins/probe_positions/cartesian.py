"""Scan patterns on a regular grid, in eight orderings and cell shapes."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import numpy
from pydantic import BaseModel, Field

from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.simulate.probe_positions import (
    ProbePositionsGenerator,
    generate_cartesian_probe_positions,
)


class CartesianVariant(IntEnum):
    """One grid, named by the three independent choices its bits carry."""

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


class EquilateralCartesianOptions(BaseModel):
    """A grid whose cells are equilateral, so the y step follows from the x step.

    There is no ``step_size_y_m`` to state: a square grid takes the x step for
    both axes, and a hexagonal one scales it by sqrt(3/4) to put each row in the
    hollow of the one before.
    """

    num_points_x: int = Field(10, ge=0, title='Number of Points X:')
    num_points_y: int = Field(10, ge=0, title='Number of Points Y:')
    step_size_x_m: float = Field(1e-6, ge=0.0, title='Step Size X:')


class CartesianOptions(EquilateralCartesianOptions):
    """A grid whose axes are stepped independently."""

    step_size_y_m: float = Field(1e-6, ge=0.0, title='Step Size Y:')


class CartesianProbePositionsGenerator(ProbePositionsGenerator):
    def __init__(self, variant: CartesianVariant) -> None:
        self._variant = variant

    def get_default_options(self) -> BaseModel:
        if self._variant.is_equilateral:
            return EquilateralCartesianOptions()

        return CartesianOptions()

    def generate(self, options: BaseModel) -> Sequence[ProbePosition]:
        if not isinstance(options, EquilateralCartesianOptions):
            raise TypeError(f'Expected EquilateralCartesianOptions, got {type(options).__name__}.')

        step_size_x_m = options.step_size_x_m

        if self._variant.is_equilateral:
            step_size_y_m = step_size_x_m

            if self._variant.is_staggered:
                step_size_y_m *= numpy.sqrt(0.75)
        elif isinstance(options, CartesianOptions):
            step_size_y_m = options.step_size_y_m
        else:
            raise TypeError(f'Expected CartesianOptions, got {type(options).__name__}.')

        return [
            *generate_cartesian_probe_positions(
                options.num_points_x,
                options.num_points_y,
                step_size_x_m,
                step_size_y_m,
                snake=self._variant.is_snaked,
                stagger=self._variant.is_staggered,
            )
        ]
