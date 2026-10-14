"""Unit tests for the APS 4-ID-B,G,H POLAR probe-position readers.

POLAR writes positions two ways -- a softGlueZynq stream for fly scans, the bluesky
``primary`` stream for step scans -- and the two carry different index conventions
against the diffraction reader. The convention tests live in
``tests/test_polar_diffraction_file.py``, where both readers are run against one scan.
"""

from __future__ import annotations

from pathlib import Path
import logging

import pytest

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.aps04id_polar_position_file import PolarPositionFileReader
from ptychodus.plugins.aps04id_polar_sgz_position_file import PolarSoftGlueZynqPositionFileReader

from polar_file_fixtures import write_eiger, write_master, write_pos_stream

MICROMETER_M = 1e-6
NANOMETER_M = 1e-9

NANOX_NANOY_YAML = '- huber_hp_nanox\n- huber_hp_nanoy\n'
# Older masters tag the sequence, which yaml.safe_load rejects outright.
NANOY_NANOZ_TUPLE_YAML = '!!python/tuple\n- huber_hp_nanoy\n- huber_hp_nanoz\n'


@pytest.fixture
def reader() -> PolarPositionFileReader:
    return PolarPositionFileReader()


def _write_fly_scan(directory: Path, *, num_triggers: int = 5) -> Path:
    """Master + eiger + pos_stream, with one detector frame per trigger but the first."""
    write_eiger(directory / 'eiger' / 'scan_000001.h5', num_patterns=num_triggers - 1)
    write_pos_stream(
        directory / 'pos_stream' / 'scan_000001.h5',
        trigger_indexes=range(num_triggers),
        x_nm=[100.0 * (index + 1) for index in range(num_triggers)],
        y_nm=[-50.0 * (index + 1) for index in range(num_triggers)],
    )
    return write_master(
        directory / 'scan_000001_master.hdf',
        plan_name='flyscan',
        eiger_filename='eiger/scan_000001.h5',
        pos_stream_filename='pos_stream/scan_000001.h5',
    )


def _write_step_scan(
    directory: Path,
    *,
    motors_yaml: str | None = NANOX_NANOY_YAML,
    primary: dict[str, list[float]] | None = None,
    num_patterns: int | None = None,
) -> Path:
    if primary is None:
        primary = {
            'huber_hp_nanox': [10.0, 11.0, 12.0, 13.0],
            'huber_hp_nanoy': [20.0, 21.0, 22.0, 23.0],
        }

    num_positions = len(next(iter(primary.values())))
    write_eiger(
        directory / 'eiger' / 'scan_000002.h5',
        num_patterns=num_positions if num_patterns is None else num_patterns,
    )
    return write_master(
        directory / 'scan_000002_master.hdf',
        plan_name='rel_grid_scan',
        eiger_filename='eiger/scan_000002.h5',
        primary=primary,
        motors_yaml=motors_yaml,
    )


def test_fly_scan_master_delegates_to_the_position_stream(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    """The master's pos_stream link is followed by filename; one position per trigger."""
    master_path = _write_fly_scan(tmp_path)

    positions = reader.read(master_path)

    assert len(positions) == 5
    assert [point.index for point in positions] == [0, 1, 2, 3, 4]
    assert positions[0].x_m == pytest.approx(-100.0 * NANOMETER_M)
    assert positions[4].y_m == pytest.approx(250.0 * NANOMETER_M)

    direct = PolarSoftGlueZynqPositionFileReader().read(tmp_path / 'pos_stream' / 'scan_000001.h5')
    assert [point.index for point in direct] == [point.index for point in positions]
    assert [point.x_m for point in direct] == [point.x_m for point in positions]


def test_fly_scan_positions_negate_the_stage_readback(tmp_path: Path) -> None:
    """Both axes are negated: the stream holds the stage readback, not probe coordinates.

    The beamline's own ``4idd_data_preprocessing_flyscan_v2.py`` applies the same flip as
    ``ppX = -xs[1:]``, ``ppY = -ys[1:]``. Pinned against the raw fixture values so a
    reader that stopped negating cannot pass.
    """
    raw_x_nm = [100.0, 200.0, 300.0]
    raw_y_nm = [-50.0, -60.0, -70.0]
    stream_path = write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000003.h5',
        trigger_indexes=range(len(raw_x_nm)),
        x_nm=raw_x_nm,
        y_nm=raw_y_nm,
    )

    positions = PolarSoftGlueZynqPositionFileReader().read(stream_path)

    assert [point.x_m for point in positions] == pytest.approx(
        [-value * NANOMETER_M for value in raw_x_nm]
    )
    assert [point.y_m for point in positions] == pytest.approx(
        [-value * NANOMETER_M for value in raw_y_nm]
    )


def test_step_scan_positions_negate_the_motor_readback(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    """The step path negates too, so the two layouts agree on the coordinate frame."""
    raw_x_um = [10.0, 11.0, 12.0, 13.0]
    raw_y_um = [20.0, 21.0, 22.0, 23.0]
    master_path = _write_step_scan(
        tmp_path,
        primary={'huber_hp_nanox': raw_x_um, 'huber_hp_nanoy': raw_y_um},
    )

    positions = reader.read(master_path)

    assert [point.x_m for point in positions] == pytest.approx(
        [-value * MICROMETER_M for value in raw_x_um]
    )
    assert [point.y_m for point in positions] == pytest.approx(
        [-value * MICROMETER_M for value in raw_y_um]
    )


def test_step_scan_master_reads_the_primary_stream(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    """Motor readbacks are micrometres; indexes come from the Eiger unique id."""
    master_path = _write_step_scan(tmp_path)

    positions = reader.read(master_path)

    assert len(positions) == 4
    assert [point.index for point in positions] == [1, 2, 3, 4]
    assert positions[0].x_m == pytest.approx(-10.0 * MICROMETER_M)
    assert positions[0].y_m == pytest.approx(-20.0 * MICROMETER_M)
    assert positions[3].x_m == pytest.approx(-13.0 * MICROMETER_M)


def test_step_scan_accepts_the_tagged_tuple_motor_list(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    """The first motor listed becomes x, whichever stage axes the scan used."""
    master_path = _write_step_scan(
        tmp_path,
        motors_yaml=NANOY_NANOZ_TUPLE_YAML,
        primary={
            'huber_hp_nanoy': [1.0, 2.0, 3.0],
            'huber_hp_nanoz': [23.0, 24.0, 25.0],
        },
    )

    positions = reader.read(master_path)

    assert positions[0].x_m == pytest.approx(-1.0 * MICROMETER_M)
    assert positions[0].y_m == pytest.approx(-23.0 * MICROMETER_M)


def test_step_scan_without_motors_metadata_falls_back(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    master_path = _write_step_scan(tmp_path, motors_yaml=None)

    positions = reader.read(master_path)

    assert positions[0].x_m == pytest.approx(-10.0 * MICROMETER_M)


def test_step_scan_naming_an_absent_motor_raises(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    master_path = _write_step_scan(tmp_path, motors_yaml='- huber_hp_nanoq\n- huber_hp_nanoy\n')

    with pytest.raises(ProbePositionParseError, match='huber_hp_nanoq'):
        reader.read(master_path)


def test_unknown_layout_names_both_layouts_and_the_plan(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    """An alignment scan carries no positions; the message has to say which."""
    write_eiger(tmp_path / 'eiger' / 'scan_000003.h5', num_patterns=3)
    master_path = write_master(
        tmp_path / 'scan_000003_master.hdf',
        plan_name='lup',
        eiger_filename='eiger/scan_000003.h5',
    )

    with pytest.raises(ProbePositionParseError) as excinfo:
        reader.read(master_path)

    message = str(excinfo.value)
    assert 'pos_stream' in message
    assert 'primary' in message
    assert 'lup' in message


def test_legacy_eiger_positions_are_refused(
    reader: PolarPositionFileReader, tmp_path: Path
) -> None:
    """Xpos/Ypos are identically zero in every real file; reading them is the bug."""
    eiger_path = write_eiger(
        tmp_path / 'eiger' / 'scan_000004.h5', num_patterns=6, legacy_positions=True
    )

    with pytest.raises(ProbePositionParseError):
        reader.read(eiger_path)


def test_non_monotonic_trigger_column_raises(tmp_path: Path) -> None:
    """reduceat grouping is only correct for a non-decreasing trigger column."""
    stream_path = write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000005.h5',
        trigger_indexes=[0, 1, 0],
        x_nm=[1.0, 2.0, 3.0],
        y_nm=[1.0, 2.0, 3.0],
    )

    with pytest.raises(ProbePositionParseError, match='non-decreasing'):
        PolarSoftGlueZynqPositionFileReader().read(stream_path)


def test_trigger_column_not_starting_at_zero_warns_but_reads(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """A free-running counter would put every pattern outside the position range."""
    stream_path = write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000006.h5',
        trigger_indexes=[9000, 9001, 9002],
        x_nm=[1.0, 2.0, 3.0],
        y_nm=[4.0, 5.0, 6.0],
    )

    with caplog.at_level(logging.WARNING):
        positions = PolarSoftGlueZynqPositionFileReader().read(stream_path)

    assert [point.index for point in positions] == [9000, 9001, 9002]
    assert 'starts at 9000' in caplog.text


def test_unexpected_column_count_warns(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    stream_path = write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000007.h5',
        trigger_indexes=[0, 1],
        x_nm=[1.0, 2.0],
        y_nm=[3.0, 4.0],
        num_columns=8,
    )

    with caplog.at_level(logging.WARNING):
        PolarSoftGlueZynqPositionFileReader().read(stream_path)

    assert 'expected 24 columns' in caplog.text


def test_trailing_plateau_rows_are_trimmed(tmp_path: Path) -> None:
    """Rows after the sample counter stalls are buffer padding, not measurements."""
    stream_path = write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000008.h5',
        trigger_indexes=[0, 1, 2],
        x_nm=[1.0, 2.0, 3.0],
        y_nm=[4.0, 5.0, 6.0],
        num_trailing_plateau_rows=5,
    )

    positions = PolarSoftGlueZynqPositionFileReader().read(stream_path)

    assert [point.index for point in positions] == [0, 1, 2]


def test_aborted_scan_falls_back_to_sequential_indexes(
    reader: PolarPositionFileReader, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    """The detector and the scan engine can stop at different points."""
    master_path = _write_step_scan(tmp_path, num_patterns=3)

    with caplog.at_level(logging.WARNING):
        positions = reader.read(master_path)

    assert [point.index for point in positions] == [1, 2, 3, 4]
    assert 'aborted scan' in caplog.text


def test_both_polar_position_readers_register() -> None:
    registry = PluginRegistry.load_plugins()
    names = {plugin.simple_name for plugin in registry.probe_position_file_readers}

    assert {'APS_Polar', 'APS_Polar_SGZ'} <= names
