"""Synthetic APS 4-ID-B,G,H POLAR HDF5 fixtures.

Shaped after the retained beamline files: a ``scan_NNNNNN_master.hdf`` wrapper that
external-links ``eiger/scan_NNNNNN.h5`` and, for fly scans, ``pos_stream/scan_NNNNNN.h5``.
Only the groups the readers actually probe are written, with detector frames shrunk to a
few pixels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import h5py
import numpy

DETECTOR_HEIGHT = 4
DETECTOR_WIDTH = 5

NUM_POS_STREAM_COLUMNS = 24
COL_I0_COUNTER = 0
COL_SAMPLE_COUNTER = 1
COL_TRIGGER = 2
COL_X = 3
COL_Y = 4

# The link target the instrument writes. It is a group, not the frame dataset.
EXTERNAL_LINK_TARGET = '/entry/instrument'


def write_eiger(
    file_path: Path,
    *,
    num_patterns: int,
    uid: Sequence[int] | None = None,
    detector_distance_mm: float | None = None,
    beam_center_px: tuple[float, float] | None = None,
    legacy_positions: bool = False,
) -> Path:
    """Write an Eiger frame file. ``uid`` defaults to a contiguous run."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    unique_id = numpy.arange(1000, 1000 + num_patterns) if uid is None else numpy.asarray(uid)

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset(
            '/entry/data/data',
            data=numpy.zeros((num_patterns, DETECTOR_HEIGHT, DETECTOR_WIDTH), dtype=numpy.int32),
        )
        attributes = h5_file.create_group('/entry/instrument/NDAttributes')
        attributes.create_dataset('NDArrayUniqueId', data=unique_id.astype(numpy.int32))

        if detector_distance_mm is not None:
            attributes.create_dataset(
                'DetectorDistance', data=numpy.full(num_patterns, detector_distance_mm)
            )
            attributes.create_dataset(
                'DistancePV', data=numpy.full(num_patterns, detector_distance_mm)
            )

        if beam_center_px is not None:
            center_x_px, center_y_px = beam_center_px
            attributes.create_dataset('BeamCenterX', data=numpy.full(num_patterns, center_x_px))
            attributes.create_dataset('BeamCenterY', data=numpy.full(num_patterns, center_y_px))
            attributes.create_dataset('BeamCenterPV', data=numpy.full(num_patterns, center_x_px))

        if legacy_positions:
            # Identically zero, exactly as every legacy file on disk has them.
            attributes.create_dataset('Xpos', data=numpy.zeros(num_patterns))
            attributes.create_dataset('Ypos', data=numpy.zeros(num_patterns))

    return file_path


def write_pos_stream(
    file_path: Path,
    *,
    trigger_indexes: Sequence[int],
    x_nm: Sequence[float],
    y_nm: Sequence[float],
    samples_per_trigger: int = 3,
    num_trailing_plateau_rows: int = 0,
    num_columns: int = NUM_POS_STREAM_COLUMNS,
) -> Path:
    """Write an oversampled softGlueZynq position stream.

    Every sample sharing a trigger carries the same coordinates, so the reader's mean is
    exact. ``num_trailing_plateau_rows`` appends rows that stall the sample counter,
    which is how the instrument pads an unfilled buffer.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[int]] = []

    for trigger, x, y in zip(trigger_indexes, x_nm, y_nm):
        for _ in range(samples_per_trigger):
            row = [0] * num_columns
            row[COL_I0_COUNTER] = 10 * len(rows)
            row[COL_SAMPLE_COUNTER] = len(rows) + 1
            row[COL_TRIGGER] = int(trigger)
            row[COL_X] = int(x)
            row[COL_Y] = int(y)
            rows.append(row)

    for _ in range(num_trailing_plateau_rows):
        row = list(rows[-1])
        # The counter stalls while the trigger keeps advancing, so a failure to trim
        # shows up as extra triggers rather than as extra samples on the last one.
        row[COL_TRIGGER] = rows[-1][COL_TRIGGER] + 1
        rows.append(row)

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('/entry/data/data', data=numpy.array(rows, dtype=numpy.int32))

    return file_path


def write_master(
    file_path: Path,
    *,
    plan_name: str | None = 'flyscan',
    eiger_filename: str | None = None,
    pos_stream_filename: str | None = None,
    primary: Mapping[str, Sequence[float]] | None = None,
    motors_yaml: str | None = None,
    mono_energy_keV: float | None = None,  # noqa: N803
) -> Path:
    """Write a master wrapper. Every section is optional so a test can omit one."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, 'w') as h5_file:
        if plan_name is not None:
            h5_file.create_dataset('/entry/plan_name', data=plan_name)

        externals = h5_file.create_group('/entry/externals')

        if eiger_filename is not None:
            externals['eiger'] = h5py.ExternalLink(eiger_filename, EXTERNAL_LINK_TARGET)

        if pos_stream_filename is not None:
            externals['pos_stream'] = h5py.ExternalLink(pos_stream_filename, EXTERNAL_LINK_TARGET)

        if primary is not None:
            stream = h5_file.create_group('/entry/instrument/bluesky/streams/primary')

            for motor, values in primary.items():
                stream.create_dataset(f'{motor}/value', data=numpy.asarray(values, dtype=float))

        if motors_yaml is not None:
            h5_file.create_dataset('/entry/instrument/bluesky/metadata/motors', data=motors_yaml)

        if mono_energy_keV is not None:
            h5_file.create_dataset(
                '/entry/instrument/bluesky/streams/baseline/mono_energy/value_start',
                data=mono_energy_keV,
            )

    return file_path
