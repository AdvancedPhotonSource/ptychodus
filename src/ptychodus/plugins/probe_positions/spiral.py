"""A Fermat spiral, each point set a golden angle round from the last."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ptychodus.api.options import expect_options
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.simulate.probe_positions import (
    ProbePositionsGenerator,
    generate_spiral_probe_positions,
)


class SpiralOptions(BaseModel):
    # One hundred because the count used to be read as a grid's two sides
    # multiplied together, and ten by ten is what that gave.
    num_points: int = Field(100, ge=0, title='Number of Points:')
    radius_scalar_m: float = Field(0.5e-6, ge=0.0, title='Radius Scalar:')


class SpiralProbePositionsGenerator(ProbePositionsGenerator):
    """https://doi.org/10.1364/OE.22.012634"""

    def get_default_options(self) -> BaseModel:
        return SpiralOptions()

    def generate(self, options: BaseModel) -> Sequence[ProbePosition]:
        spiral = expect_options(options, SpiralOptions)
        return [*generate_spiral_probe_positions(spiral.num_points, spiral.radius_scalar_m)]
