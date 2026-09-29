"""Unit tests for the APS 19-ID-E In-situ Nanoprobe probe-position reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy
import pytest

from ptychodus.api.constants import LengthUnit
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.aps19id_isn_position_file import ISNPositionFileReader


@pytest.fixture
def reader() -> ISNPositionFileReader:
    return ISNPositionFileReader()


def _write_position_file(
    path: Path, x_um: numpy.ndarray, y_um: numpy.ndarray, *, trigger: numpy.ndarray | None = None
) -> Path:
    with h5py.File(path, 'w') as h5_file:
        h5_file.create_dataset('/entry/data/X_Position', data=x_um)
        h5_file.create_dataset('/entry/data/Y_Position', data=y_um)

        if trigger is not None:
            h5_file.create_dataset('/entry/data/Trigger', data=trigger)

    return path


def test_normal_read_converts_units_and_indexes_by_array_order(
    reader: ISNPositionFileReader, tmp_path: Path
) -> None:
    x_um = numpy.array([1.0, 2.0, 3.0])
    y_um = numpy.array([-1.0, -2.0, -3.0])
    # Trigger deliberately does not start at 0 or run sequentially, so a reader that
    # indexed by it instead of array order would be caught here.
    trigger = numpy.array([7, 7, 9])
    file_path = _write_position_file(
        tmp_path / 'Scan_0001_position.h5', x_um, y_um, trigger=trigger
    )

    positions = reader.read(file_path)

    assert len(positions) == 3
    assert [point.index for point in positions] == [0, 1, 2]

    for point, x, y in zip(positions, x_um, y_um):
        assert point.x_m == pytest.approx(LengthUnit.MICROMETER.to_meters(float(x)))
        assert point.y_m == pytest.approx(LengthUnit.MICROMETER.to_meters(float(y)))


def test_shape_mismatch_is_rejected(reader: ISNPositionFileReader, tmp_path: Path) -> None:
    file_path = _write_position_file(
        tmp_path / 'Scan_0002_position.h5', numpy.array([1.0, 2.0]), numpy.array([1.0])
    )

    with pytest.raises(ProbePositionParseError, match='shape mismatch'):
        reader.read(file_path)


def test_missing_dataset_is_rejected(reader: ISNPositionFileReader, tmp_path: Path) -> None:
    file_path = tmp_path / 'Scan_0003_position.h5'

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('/entry/data/X_Position', data=numpy.array([1.0, 2.0]))

    with pytest.raises(ProbePositionParseError, match='Missing position dataset'):
        reader.read(file_path)
