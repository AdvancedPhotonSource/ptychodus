from __future__ import annotations
from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sequence
import logging

import numpy

from ptychodus.api.geometry import AffineTransform
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionSequence,
)
from ptychodus.api.probe_positions_gen import transform_probe_positions

from .settings import ProbePositionsSettings

logger = logging.getLogger(__name__)


class ProbePositionsBuilder(ParameterGroup):
    def __init__(
        self, rng: numpy.random.Generator, settings: ProbePositionsSettings, name: str
    ) -> None:
        super().__init__()
        self._rng = rng

        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

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

    def get_name(self) -> str:
        return self._name.get_value()

    @staticmethod
    def negate_x(preset: int) -> bool:
        return preset & 0x1 != 0x0

    @staticmethod
    def negate_y(preset: int) -> bool:
        return preset & 0x2 != 0x0

    @staticmethod
    def swap_xy(preset: int) -> bool:
        return preset & 0x4 != 0x0

    def labels_for_preset_transforms(self) -> Iterator[str]:
        for index in range(8):
            xp = '\u2212x' if self.negate_x(index) else '\u002bx'
            yp = '\u2212y' if self.negate_y(index) else '\u002by'
            fxy = f'{yp}, {xp}' if self.swap_xy(index) else f'{xp}, {yp}'
            yield f'(x, y) \u2192 ({fxy})'

    def assign_preset_transform(self, index: int) -> None:
        self.block_notifications(True)

        if self.swap_xy(index):
            self.affine00.set_value(0)
            self.affine01.set_value(-1 if self.negate_y(index) else +1)
            self.affine02.set_value(0)
            self.affine10.set_value(-1 if self.negate_x(index) else +1)
            self.affine11.set_value(0)
            self.affine12.set_value(0)
        else:
            self.affine00.set_value(-1 if self.negate_x(index) else +1)
            self.affine01.set_value(0)
            self.affine02.set_value(0)
            self.affine10.set_value(0)
            self.affine11.set_value(-1 if self.negate_y(index) else +1)
            self.affine12.set_value(0)

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

    def set_transform(self, transform: AffineTransform) -> None:
        self.block_notifications(True)

        self.affine00.set_value(transform.a00)
        self.affine01.set_value(transform.a01)
        self.affine02.set_value(transform.a02)

        self.affine10.set_value(transform.a10)
        self.affine11.set_value(transform.a11)
        self.affine12.set_value(transform.a12)

        self.block_notifications(False)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    @abstractmethod
    def copy(self) -> ProbePositionsBuilder:
        pass

    @abstractmethod
    def build(self) -> ProbePositionSequence:
        pass

    def _create_position_sequence(
        self, positions: Iterable[ProbePosition]
    ) -> ProbePositionSequence:
        transform = self.get_transform()
        jitter_radius_m = self.jitter_radius_m.get_value()
        rng = self._rng if jitter_radius_m > 0.0 else None
        return ProbePositionSequence(
            [*transform_probe_positions(positions, transform, rng, jitter_radius_m)]
        )


class FromMemoryProbePositionsBuilder(ProbePositionsBuilder):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        position_seq: Sequence[ProbePosition],
    ) -> None:
        super().__init__(rng, settings, 'from_memory')
        self._rng = rng
        self._settings = settings
        self._position_seq = ProbePositionSequence(position_seq)

        # set identity transformation
        self.assign_preset_transform(0)

    def copy(self) -> FromMemoryProbePositionsBuilder:
        builder = FromMemoryProbePositionsBuilder(self._rng, self._settings, self._position_seq)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> ProbePositionSequence:
        return self._position_seq


class FromFileProbePositionsBuilder(ProbePositionsBuilder):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        file_reader: ProbePositionFileReader,
    ) -> None:
        super().__init__(rng, settings, 'from_file')
        self._rng = rng
        self._settings = settings
        self._file_reader = file_reader

        self.file_path = settings.file_path.copy()
        self._add_parameter('file_path', self.file_path)

        self.file_type = settings.file_type.copy()
        self._add_parameter('file_type', self.file_type)

    def copy(self) -> FromFileProbePositionsBuilder:
        builder = FromFileProbePositionsBuilder(self._rng, self._settings, self._file_reader)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self) -> ProbePositionSequence:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            position_seq = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        return position_seq
