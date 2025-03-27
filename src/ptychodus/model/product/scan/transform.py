from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import ScanPoint

from .settings import ScanSettings


class ScanPointTransform(ParameterGroup):
    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings

        self.affine_ax = settings.affine_transform_ax.copy()
        self._add_parameter('affine_ax', self.affine_ax)

        self.affine_ay = settings.affine_transform_ay.copy()
        self._add_parameter('affine_ay', self.affine_ay)

        self.affine_at_m = settings.affine_transform_at_m.copy()
        self._add_parameter('affine_at_m', self.affine_at_m)

        self.affine_bx = settings.affine_transform_bx.copy()
        self._add_parameter('affine_bx', self.affine_bx)

        self.affine_by = settings.affine_transform_by.copy()
        self._add_parameter('affine_by', self.affine_by)

        self.affine_bt_m = settings.affine_transform_bt_m.copy()
        self._add_parameter('affine_bt_m', self.affine_bt_m)

        self.jitter_radius_m = settings.jitter_radius_m.copy()
        self._add_parameter('jitter_radius_m', self.jitter_radius_m)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def copy(self) -> ScanPointTransform:
        transform = ScanPointTransform(self._rng, self._settings)

        for key, value in self.parameters().items():
            transform.parameters()[key].set_value(value.get_value())

        return transform

    @staticmethod
    def negate_x(preset: int) -> bool:
        return preset & 0x1 != 0x0

    @staticmethod
    def negate_y(preset: int) -> bool:
        return preset & 0x2 != 0x0

    @staticmethod
    def swap_xy(preset: int) -> bool:
        return preset & 0x4 != 0x0

    def labels_for_presets(self) -> Iterator[str]:
        for index in range(8):
            xp = '\u2212x' if self.negate_x(index) else '\u002bx'
            yp = '\u2212y' if self.negate_y(index) else '\u002by'
            fxy = f'{yp}, {xp}' if self.swap_xy(index) else f'{xp}, {yp}'
            yield f'(x, y) \u2192 ({fxy})'

    def apply_presets(self, index: int) -> None:
        if self.swap_xy(index):
            self.affine_ay.set_value(-1 if self.negate_y(index) else +1)
            self.affine_bx.set_value(-1 if self.negate_x(index) else +1)
            self.affine_ax.set_value(0)
            self.affine_by.set_value(0)
        else:
            self.affine_ax.set_value(-1 if self.negate_x(index) else +1)
            self.affine_by.set_value(-1 if self.negate_y(index) else +1)
            self.affine_ay.set_value(0)
            self.affine_bx.set_value(0)

    def set_identity(self) -> None:
        self.apply_presets(0)

    def __call__(self, point: ScanPoint) -> ScanPoint:
        ax = self.affine_ax.get_value()
        ay = self.affine_ay.get_value()
        at_m = self.affine_at_m.get_value()

        bx = self.affine_bx.get_value()
        by = self.affine_by.get_value()
        bt_m = self.affine_bt_m.get_value()

        pos_x = ax * point.position_x_m + ay * point.position_y_m + at_m
        pos_y = bx * point.position_x_m + by * point.position_y_m + bt_m

        rad = self.jitter_radius_m.get_value()

        if rad > 0.0:
            while True:
                dx = self._rng.uniform()
                dy = self._rng.uniform()

                if dx * dx + dy * dy < 1.0:
                    pos_x += dx * rad
                    pos_y += dy * rad
                    break

        return ScanPoint(point.index, pos_x, pos_y)
