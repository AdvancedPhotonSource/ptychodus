from __future__ import annotations
from abc import abstractmethod
from collections.abc import Iterator, Sequence
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

        self.num_discard_at_start = settings.num_discard_at_start.copy()
        self._add_parameter('num_discard_at_start', self.num_discard_at_start)

        self.num_discard_at_end = settings.num_discard_at_end.copy()
        self._add_parameter('num_discard_at_end', self.num_discard_at_end)

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
    def _build_raw(self) -> Sequence[ProbePosition]:
        """Return the raw, unconditioned probe positions in acquisition order.

        Implementations must NOT apply the trim, affine transform, or jitter;
        `build` owns the conditioning pipeline.
        """
        pass

    def build(self) -> ProbePositionSequence:
        """Return the conditioned probe positions: trim, then affine, then jitter.

        Overriding this method is reserved for builders whose positions are
        already conditioned; see `FromMemoryProbePositionsBuilder`. Every builder
        that ingests raw instrument coordinates must leave it alone and implement
        `_build_raw` instead.
        """
        return self._condition_positions(self._build_raw())

    def _condition_positions(self, positions: Sequence[ProbePosition]) -> ProbePositionSequence:
        trimmed = self._trim_positions(positions)
        transform = self.get_transform()
        jitter_radius_m = self.jitter_radius_m.get_value()
        rng = self._rng if jitter_radius_m > 0.0 else None
        return ProbePositionSequence(
            [*transform_probe_positions(trimmed, transform, rng, jitter_radius_m)]
        )

    def _trim_positions(self, positions: Sequence[ProbePosition]) -> Sequence[ProbePosition]:
        """Discard points from each end of the scan, in acquisition order.

        Surviving points keep their original scan indexes. Diffraction patterns
        whose index falls outside the trimmed range are dropped downstream by
        `AssembledDiffractionData.prepare_reconstruct_input`, which never
        extrapolates beyond the position-index anchors.
        """
        num_discard_at_start = self.num_discard_at_start.get_value()
        num_discard_at_end = self.num_discard_at_end.get_value()

        if num_discard_at_start == 0 and num_discard_at_end == 0:
            return positions

        num_positions = len(positions)
        stop = num_positions - num_discard_at_end

        if stop <= num_discard_at_start:
            logger.warning(
                f'Discarding {num_discard_at_start} probe position(s) at the start and'
                f' {num_discard_at_end} at the end leaves nothing of {num_positions}!'
            )
            return []

        return positions[num_discard_at_start:stop]


class FromMemoryProbePositionsBuilder(ProbePositionsBuilder):
    """Probe positions that have already been conditioned.

    Two things produce these. Reconstruction output, which `ProcessingTaskMonitor`
    re-assigns to the output product item on every reconstructor iteration (see
    `model/processing/monitor.py`), and products loaded from HDF5/NPZ, whose
    positions were conditioned before they were saved. In both cases the trim,
    affine transform, and jitter have already been applied upstream. Re-applying
    them here would corrupt position-corrected output a little more on every
    iteration, and would move the positions out of the coordinate frame the
    reconstructed object was solved in. `build` therefore deliberately bypasses
    the conditioning pipeline.
    """

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

    def copy(self) -> FromMemoryProbePositionsBuilder:
        builder = FromMemoryProbePositionsBuilder(self._rng, self._settings, self._position_seq)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self) -> ProbePositionSequence:
        return self._position_seq

    def build(self) -> ProbePositionSequence:
        return self._build_raw()


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

    def _build_raw(self) -> ProbePositionSequence:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            position_seq = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        return position_seq
