from __future__ import annotations

import numpy

from ptychodus.api.scan import PositionSequence, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class LissajousScanBuilder(ScanBuilder):
    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'lissajous')
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

    def copy(self) -> LissajousScanBuilder:
        builder = LissajousScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        for index in range(self.num_points.get_value()):
            two_pi = 2 * numpy.pi
            theta_x = (
                two_pi * self.angular_step_x_turns.get_value() * index
                + self.angular_shift_turns.get_value()
            )
            theta_y = two_pi * self.angular_step_y_turns.get_value() * index

            point = ScanPoint(
                index=index,
                position_x_m=self.amplitude_x_m.get_value() * numpy.sin(theta_x),
                position_y_m=self.amplitude_y_m.get_value() * numpy.sin(theta_y),
            )
            point_list.append(point)

        return PositionSequence(point_list)
