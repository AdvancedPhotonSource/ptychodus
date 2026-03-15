from __future__ import annotations

import numpy

from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.probe_positions_gen import generate_lissajous_probe_positions

from .builder import ProbePositionsBuilder
from .settings import ProbePositionsSettings


class LissajousProbePositionsBuilder(ProbePositionsBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbePositionsSettings) -> None:
        super().__init__(rng, settings, 'lissajous')
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

        self.amplitude_x_m = settings.amplitude_x_m.copy()
        self._add_parameter('amplitude_x_m', self.amplitude_x_m)

        self.amplitude_y_m = settings.amplitude_y_m.copy()
        self._add_parameter('amplitude_y_m', self.amplitude_y_m)

        self.angular_step_x_turns = settings.angular_step_x_turns.copy()
        self._add_parameter('angular_step_x_tr', self.angular_step_x_turns)

        self.angular_step_y_turns = settings.angular_step_y_turns.copy()
        self._add_parameter('angular_step_y_tr', self.angular_step_y_turns)

        self.angular_shift_turns = settings.angular_shift_turns.copy()
        self._add_parameter('angular_shift_tr', self.angular_shift_turns)

    def copy(self) -> LissajousProbePositionsBuilder:
        builder = LissajousProbePositionsBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> ProbePositionSequence:
        positions = generate_lissajous_probe_positions(
            self.num_points.get_value(),
            self.amplitude_x_m.get_value(),
            self.amplitude_y_m.get_value(),
            self.angular_step_x_turns.get_value(),
            self.angular_step_y_turns.get_value(),
            self.angular_shift_turns.get_value(),
        )
        return self._create_position_sequence(positions)
