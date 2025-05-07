from __future__ import annotations

import numpy

from ptychodus.api.scan import PositionSequence, ScanPoint

from .builder import ScanBuilder
from .settings import ScanSettings


class SpiralScanBuilder(ScanBuilder):
    """https://doi.org/10.1364/OE.22.012634"""

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__(settings, 'spiral')
        self._settings = settings

        self.num_points = settings.num_points_x.copy()
        self.num_points.set_value(
            settings.num_points_x.get_value() * settings.num_points_y.get_value()
        )
        self._add_parameter('num_points', self.num_points)

        self._num_points = settings.num_points_y.copy()
        self._num_points.set_value(1)
        self._add_parameter('_num_points', self._num_points)

        self.radius_scalar_m = settings.radius_scalar_m.copy()
        self._add_parameter('radius_scalar_m', self.radius_scalar_m)

    def copy(self) -> SpiralScanBuilder:
        builder = SpiralScanBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        for index in range(self.num_points.get_value()):
            radius_m = self.radius_scalar_m.get_value() * numpy.sqrt(index)
            divergence_angle_rad = (3.0 - numpy.sqrt(5)) * numpy.pi
            theta_rad = divergence_angle_rad * index

            point = ScanPoint(
                index=index,
                position_x_m=radius_m * numpy.cos(theta_rad),
                position_y_m=radius_m * numpy.sin(theta_rad),
            )
            point_list.append(point)

        return PositionSequence(point_list)
