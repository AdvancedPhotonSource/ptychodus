from __future__ import annotations

import numpy

from ptychodus.api.scan import PositionSequence, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class ConcentricScanBuilder(ScanBuilder):
    """https://doi.org/10.1088/1367-2630/12/3/035017"""

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'concentric')
        self._settings = settings

        self.radial_step_size_m = settings.radial_step_size_m.copy()
        self._add_parameter('radial_step_size_m', self.radial_step_size_m)

        self.num_shells = settings.num_shells.copy()
        self._add_parameter('num_shells', self.num_shells)

        self.num_points_1st_shell = settings.num_points_in_first_shell.copy()
        self._add_parameter('num_points_1st_shell', self.num_points_1st_shell)

    def copy(self) -> ConcentricScanBuilder:
        builder = ConcentricScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def _num_points(self) -> int:
        num_shells = self.num_shells.get_value()
        triangle = (num_shells * (num_shells + 1)) // 2
        return triangle * self.num_points_1st_shell.get_value()

    def build(self) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        for index in range(self._num_points):
            triangle = index // self.num_points_1st_shell.get_value()
            shell_index = int((1 + numpy.sqrt(1 + 8 * triangle)) / 2) - 1  # see OEIS A002024
            shell_triangle = (shell_index * (shell_index + 1)) // 2
            first_index_in_shell = self.num_points_1st_shell.get_value() * shell_triangle
            point_index_in_shell = index - first_index_in_shell

            radius_m = self.radial_step_size_m.get_value() * (shell_index + 1)
            num_points_in_shell = self.num_points_1st_shell.get_value() * (shell_index + 1)
            theta_rad = 2 * numpy.pi * point_index_in_shell / num_points_in_shell

            point = ScanPoint(
                index=index,
                position_x_m=radius_m * numpy.cos(theta_rad),
                position_y_m=radius_m * numpy.sin(theta_rad),
            )
            point_list.append(point)

        return PositionSequence(point_list)
