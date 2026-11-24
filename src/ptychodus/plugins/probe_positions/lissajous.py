"""A Lissajous figure, traced by two sinusoids of incommensurate period."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ptychodus.api.options import expect_options
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.simulate.probe_positions import (
    ProbePositionsGenerator,
    generate_lissajous_probe_positions,
)


class LissajousOptions(BaseModel):
    # Shares NumberOfPoints with the spiral, so the two must agree on a default.
    num_points: int = Field(100, ge=0, title='Number of Points:')
    amplitude_x_m: float = Field(4.5e-6, ge=0.0, title='Amplitude X:')
    amplitude_y_m: float = Field(4.5e-6, ge=0.0, title='Amplitude Y:')
    # Unbounded, and so signed: a negative step reverses the sweep.
    angular_step_x_deg: float = Field(10.8, title='Angular Step X:')
    angular_step_y_deg: float = Field(14.4, title='Angular Step Y:')
    angular_shift_deg: float = Field(90.0, title='Angular Shift:')


class LissajousProbePositionsGenerator(ProbePositionsGenerator):
    def get_default_options(self) -> BaseModel:
        return LissajousOptions()

    def generate(self, options: BaseModel) -> Sequence[ProbePosition]:
        lissajous = expect_options(options, LissajousOptions)
        return [
            *generate_lissajous_probe_positions(
                lissajous.num_points,
                lissajous.amplitude_x_m,
                lissajous.amplitude_y_m,
                lissajous.angular_step_x_deg,
                lissajous.angular_step_y_deg,
                lissajous.angular_shift_deg,
            )
        ]
