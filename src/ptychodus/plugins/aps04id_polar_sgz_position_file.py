from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.constants import LengthUnit
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)

DATA_PATH: Final[str] = '/entry/data/data'
COL_I0_COUNTER: Final[int] = 0
COL_SAMPLE_COUNTER: Final[int] = 1
COL_TRIGGER: Final[int] = 2
COL_X: Final[int] = 3
COL_Y: Final[int] = 4
EXPECTED_NUM_COLUMNS: Final[int] = 24


def read_position_stream(file_path: Path) -> ProbePositionSequence:
    """Aggregate a softGlueZynq position stream into one position per trigger.

    The stream is oversampled relative to the detector: many raw samples share a
    single trigger index. Grouping matches ``process_flyscan.process_position_stream``.

    The emitted index is the **raw** trigger counter, which the instrument starts at
    zero, while the diffraction reader emits 1-based frame indexes. Frame ``k`` therefore
    pairs with trigger ``k + 1`` and trigger ``0`` is left over -- the offset
    ``process_flyscan.plot_data`` implements as ``x[1:]``. Do not shift here.
    """
    with h5py.File(file_path, 'r') as h5_file:
        try:
            pos_raw = h5_file[DATA_PATH][()]
        except KeyError as ex:
            raise ProbePositionParseError(f'Missing dataset {DATA_PATH!r}.') from ex

        if pos_raw.ndim != 2 or pos_raw.shape[1] <= max(COL_X, COL_Y):
            raise ProbePositionParseError(
                f'Unexpected dataset shape {pos_raw.shape} at {DATA_PATH}.'
            )
        if pos_raw.shape[0] < 2:
            raise ProbePositionParseError(
                f'Need at least 2 samples at {DATA_PATH}; got {pos_raw.shape[0]}.'
            )

        num_columns = pos_raw.shape[1]
        logger.debug(f'{file_path.name}: {num_columns} columns at {DATA_PATH}.')

        if num_columns != EXPECTED_NUM_COLUMNS:
            logger.warning(
                f'{file_path.name}: expected {EXPECTED_NUM_COLUMNS} columns at {DATA_PATH}, '
                f'found {num_columns}. The column assignment (trigger={COL_TRIGGER}, '
                f'x={COL_X}, y={COL_Y}) may no longer be correct.'
            )

        counter = pos_raw[:, COL_SAMPLE_COUNTER]
        plateau = numpy.where(numpy.diff(counter) == 0)[0]
        n_valid = int(plateau[0]) + 1 if plateau.size else pos_raw.shape[0]
        pos = pos_raw[:n_valid]

        trig = pos[:, COL_TRIGGER]

        if numpy.any(numpy.diff(trig) < 0):
            raise ProbePositionParseError(
                f'Trigger column {COL_TRIGGER} at {DATA_PATH} is not non-decreasing; '
                'grouping by trigger would silently interleave unrelated samples.'
            )

        starts = numpy.concatenate(([0], numpy.where(numpy.diff(trig) != 0)[0] + 1))
        counts = numpy.diff(numpy.concatenate((starts, [len(trig)])))
        trigger_indexes = trig[starts].astype(int)

        if trigger_indexes[0] != 0:
            logger.warning(
                f'{file_path.name}: trigger column starts at {trigger_indexes[0]}, not 0. '
                'The diffraction reader emits 1-based frame indexes, so positions are '
                'expected to run 0..N. A large offset means a free-running trigger counter, '
                'which puts every pattern outside the position-index range and makes '
                'prepare_reconstruct_input drop the whole scan.'
            )

        xs = numpy.add.reduceat(pos[:, COL_X], starts) / counts
        ys = numpy.add.reduceat(pos[:, COL_Y], starts) / counts

        dxs = numpy.maximum.reduceat(pos[:, COL_X], starts) - numpy.minimum.reduceat(
            pos[:, COL_X], starts
        )
        dys = numpy.maximum.reduceat(pos[:, COL_Y], starts) - numpy.minimum.reduceat(
            pos[:, COL_Y], starts
        )
        i0s = numpy.maximum.reduceat(pos[:, COL_I0_COUNTER], starts) - numpy.minimum.reduceat(
            pos[:, COL_I0_COUNTER], starts
        )
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
            LengthUnit.NANOMETER.to_meters(float(x)),
            LengthUnit.NANOMETER.to_meters(float(y)),
            probe_photon_count=float(i0),
        )
        for trigger_index, x, y, i0 in zip(trigger_indexes, xs, ys, i0s)
    ]
    return ProbePositionSequence(point_list)


class PolarSoftGlueZynqPositionFileReader(ProbePositionFileReader):
    """Reader for APS 4-ID-B,G,H POLAR softGlueZynq position-stream files.

    A thin wrapper over ``read_position_stream``, which the master-file reader also
    calls when a master advertises a ``pos_stream`` external link.
    """

    def read(self, file_path: Path) -> ProbePositionSequence:
        return read_position_stream(file_path)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        PolarSoftGlueZynqPositionFileReader(),
        simple_name='APS_Polar_SGZ',
        display_name='APS 4-ID-B,G,H POLAR softGlueZynq Files (*.h5)',
    )
