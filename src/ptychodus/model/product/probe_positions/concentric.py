from __future__ import annotations

import numpy

from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.probe_positions_gen import generate_concentric_probe_positions

from .builder import ProbePositionsBuilder
from .settings import ProbePositionsSettings


class ConcentricProbePositionsBuilder(ProbePositionsBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbePositionsSettings) -> None:
        super().__init__(rng, settings, 'concentric')
        self._settings = settings

        self.radial_step_size_m = settings.radial_step_size_m.copy()
        self._add_parameter('radial_step_size_m', self.radial_step_size_m)

        self.num_shells = settings.num_shells.copy()
        self._add_parameter('num_shells', self.num_shells)

        self.num_points_1st_shell = settings.num_points_in_first_shell.copy()
        self._add_parameter('num_points_1st_shell', self.num_points_1st_shell)

    def copy(self) -> ConcentricProbePositionsBuilder:
        builder = ConcentricProbePositionsBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> ProbePositionSequence:
        positions = generate_concentric_probe_positions(
            self.radial_step_size_m.get_value(),
            self.num_shells.get_value(),
            self.num_points_1st_shell.get_value(),
        )
        return self._create_position_sequence(positions)
