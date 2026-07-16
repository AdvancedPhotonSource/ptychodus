from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class PolarSoftGlueZynqPositionFileReader(ProbePositionFileReader):
    """Reader for APS 4-ID Polar softGlueZynq position-stream files.

    The pos_stream is oversampled relative to the detector: many raw
    samples share a single trigger index. This reader aggregates by
    trigger (matching ``process_flyscan.process_position_stream``) and
    emits one ProbePosition per trigger, indexed by the trigger counter
    from column 2.
    """

    ONE_NANOMETER_M: Final[float] = 1.0e-9
    DATA_PATH: Final[str] = '/entry/data/data'
    COL_I0_COUNTER: Final[int] = 0
    COL_SAMPLE_COUNTER: Final[int] = 1
    COL_TRIGGER: Final[int] = 2
    COL_X: Final[int] = 3
    COL_Y: Final[int] = 4

    def read(self, file_path: Path) -> ProbePositionSequence:
        with h5py.File(file_path, 'r') as h5_file:
            try:
                pos_raw = h5_file[self.DATA_PATH][()]
            except KeyError as ex:
                raise ProbePositionParseError(f'Missing dataset {self.DATA_PATH!r}.') from ex

            if pos_raw.ndim != 2 or pos_raw.shape[1] <= max(self.COL_X, self.COL_Y):
                raise ProbePositionParseError(
                    f'Unexpected dataset shape {pos_raw.shape} at {self.DATA_PATH}.'
                )
            if pos_raw.shape[0] < 2:
                raise ProbePositionParseError(
                    f'Need at least 2 samples at {self.DATA_PATH}; got {pos_raw.shape[0]}.'
                )

            counter = pos_raw[:, self.COL_SAMPLE_COUNTER]
            plateau = numpy.where(numpy.diff(counter) == 0)[0]
            n_valid = int(plateau[0]) + 1 if plateau.size else pos_raw.shape[0]
            pos = pos_raw[:n_valid]

            trig = pos[:, self.COL_TRIGGER]
            starts = numpy.concatenate(([0], numpy.where(numpy.diff(trig) != 0)[0] + 1))
            counts = numpy.diff(numpy.concatenate((starts, [len(trig)])))
            trigger_indexes = trig[starts].astype(int)

            xs = numpy.add.reduceat(pos[:, self.COL_X], starts) / counts
            ys = numpy.add.reduceat(pos[:, self.COL_Y], starts) / counts

            dxs = numpy.maximum.reduceat(pos[:, self.COL_X], starts) - numpy.minimum.reduceat(
                pos[:, self.COL_X], starts
            )
            dys = numpy.maximum.reduceat(pos[:, self.COL_Y], starts) - numpy.minimum.reduceat(
                pos[:, self.COL_Y], starts
            )
            i0s = numpy.maximum.reduceat(
                pos[:, self.COL_I0_COUNTER], starts
            ) - numpy.minimum.reduceat(pos[:, self.COL_I0_COUNTER], starts)
            logger.debug(
                f'{file_path.name}: {len(trigger_indexes)} triggers from '
                f'{n_valid}/{pos_raw.shape[0]} rows (trailing plateau trimmed: '
                f'{pos_raw.shape[0] - n_valid}). '
                f'x-jitter mean/max = {dxs.mean():.1f}/{dxs.max()} nm; '
                f'y-jitter mean/max = {dys.mean():.1f}/{dys.max()} nm; '
                f'I0 delta mean/min/max = {i0s.mean():.1f}/{i0s.min()}/{i0s.max()}.'
            )

        point_list = [
            ProbePosition(
                int(trigger_index),
                float(x) * self.ONE_NANOMETER_M,
                float(y) * self.ONE_NANOMETER_M,
            )
            for trigger_index, x, y in zip(trigger_indexes, xs, ys)
        ]
        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        PolarSoftGlueZynqPositionFileReader(),
        simple_name='APS_Polar_SGZ',
        display_name='APS 4-ID Polar softGlueZynq Files (*.h5)',
    )
