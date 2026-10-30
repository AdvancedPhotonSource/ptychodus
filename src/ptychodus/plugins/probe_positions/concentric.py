"""A scan on concentric shells, each holding more points than the one inside it."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.simulate.probe_positions import (
    ProbePositionsGenerator,
    generate_concentric_probe_positions,
)


class ConcentricOptions(BaseModel):
    radial_step_size_m: float = Field(1e-6, ge=0.0, title='Radial Step Size:')
    num_shells: int = Field(5, ge=0, title='Number of Shells:')
    num_points_in_first_shell: int = Field(10, ge=0, title='Number of Points in First Shell:')


class ConcentricProbePositionsGenerator(ProbePositionsGenerator):
    """https://doi.org/10.1088/1367-2630/12/3/035017"""

    def get_default_options(self) -> BaseModel:
        return ConcentricOptions()

    def generate(self, options: BaseModel) -> Sequence[ProbePosition]:
        if not isinstance(options, ConcentricOptions):
            raise TypeError(f'Expected ConcentricOptions, got {type(options).__name__}.')

        return [
            *generate_concentric_probe_positions(
                options.radial_step_size_m,
                options.num_shells,
                options.num_points_in_first_shell,
            )
        ]
