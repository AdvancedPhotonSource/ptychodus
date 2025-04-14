from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.geometry import AffineTransform
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import ScanPoint

from .settings import ScanSettings


class ScanPointTransform(ParameterGroup):
    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings

        self.affine00 = settings.affine00.copy()
        self._add_parameter('affine00', self.affine00)

        self.affine01 = settings.affine01.copy()
        self._add_parameter('affine01', self.affine01)

        self.affine02 = settings.affine02.copy()
        self._add_parameter('affine02', self.affine02)

        self.affine10 = settings.affine10.copy()
        self._add_parameter('affine10', self.affine10)

        self.affine11 = settings.affine11.copy()
        self._add_parameter('affine11', self.affine11)

        self.affine12 = settings.affine12.copy()
        self._add_parameter('affine12', self.affine12)

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
            yp = '\u2212y' if self.negate_y(index) else '\u002by'
            xp = '\u2212x' if self.negate_x(index) else '\u002bx'
            fyx = f'{xp}, {yp}' if self.swap_xy(index) else f'{yp}, {xp}'
            yield f'(y, x) \u2192 ({fyx})'

    def apply_presets(self, index: int) -> None:
        self.block_notifications(True)

        if self.swap_xy(index):
            self.affine00.set_value(0)
            self.affine01.set_value(-1 if self.negate_x(index) else +1)
            self.affine10.set_value(-1 if self.negate_y(index) else +1)
            self.affine11.set_value(0)
        else:
            self.affine00.set_value(-1 if self.negate_y(index) else +1)
            self.affine01.set_value(0)
            self.affine10.set_value(0)
            self.affine11.set_value(-1 if self.negate_x(index) else +1)

        self.block_notifications(False)

    def get_transform(self) -> AffineTransform:
        return AffineTransform(
            a00=self.affine00.get_value(),
            a01=self.affine01.get_value(),
            a02=self.affine02.get_value(),
            a10=self.affine10.get_value(),
            a11=self.affine11.get_value(),
            a12=self.affine12.get_value(),
        )

    def set_identity(self) -> None:
        self.apply_presets(0)

    def __call__(self, point: ScanPoint) -> ScanPoint:
        transform = self.get_transform()
        pos_y, pos_x = transform(point.position_y_m, point.position_x_m)
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
