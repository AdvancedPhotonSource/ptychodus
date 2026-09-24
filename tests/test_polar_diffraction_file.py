"""Unit tests for the APS 4-ID-B,G,H POLAR diffraction reader.

The last two tests are the point of the module: POLAR pairs patterns with positions by
scan index, and the two scan types pair by *different* conventions. Both are pinned here,
against the position readers, because a change that "fixes" one into agreement with the
other would break it silently -- the symptom is an object that never converges, not an
exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.api.diffraction import BeamCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.plugins.aps04id_polar_diffraction_file import PolarDiffractionFileReader
from ptychodus.plugins.aps04id_polar_position_file import PolarPositionFileReader

from polar_file_fixtures import write_eiger, write_master, write_pos_stream

NANOMETER_M = 1e-9

DETECTOR_DISTANCE_MM = 1909.999745
BEAM_CENTER_PX = (709.0, 378.0)
MONO_ENERGY_KEV = 6.204927737383656


@pytest.fixture
def reader() -> PolarDiffractionFileReader:
    return PolarDiffractionFileReader()


def _write_new_format(tmp_path: Path, *, num_patterns: int = 4) -> Path:
    write_eiger(
        tmp_path / 'eiger' / 'scan_000450.h5',
        num_patterns=num_patterns,
        detector_distance_mm=DETECTOR_DISTANCE_MM,
        beam_center_px=BEAM_CENTER_PX,
    )
    return write_master(
        tmp_path / 'scan_000450_master.hdf',
        plan_name='flyscan',
        eiger_filename='eiger/scan_000450.h5',
        mono_energy_keV=MONO_ENERGY_KEV,
    )


def test_new_format_metadata_is_read(reader: PolarDiffractionFileReader, tmp_path: Path) -> None:
    master_path = _write_new_format(tmp_path)

    metadata = reader.read(master_path).get_metadata()

    assert metadata.detector_distance_m == pytest.approx(1.909999745)
    assert metadata.beam_center == BeamCenter(709, 378)
    assert metadata.probe_energy_eV == pytest.approx(6204.927737383656)


def test_detector_pixel_geometry_is_the_eiger_pitch(
    reader: PolarDiffractionFileReader, tmp_path: Path
) -> None:
    """No POLAR layout records a pixel pitch, so the reader supplies the Eiger's own.

    Without it ``metadata.detector_pixel_geometry`` is None, and a caller then has no
    way to derive a probe geometry without being told the pitch by hand.
    """
    master_path = _write_new_format(tmp_path)

    pixel_geometry = reader.read(master_path).get_metadata().detector_pixel_geometry

    assert pixel_geometry is not None
    assert pixel_geometry.width_m == pytest.approx(75e-6)
    assert pixel_geometry.height_m == pytest.approx(75e-6)


def test_old_format_yields_energy_without_detector_attributes(
    reader: PolarDiffractionFileReader, tmp_path: Path
) -> None:
    """mono_energy lives on the master's baseline stream, which both formats have."""
    write_eiger(tmp_path / 'eiger' / 'scan_001634.h5', num_patterns=4)
    master_path = write_master(
        tmp_path / 'scan_001634_master.hdf',
        plan_name='rel_grid_scan',
        eiger_filename='eiger/scan_001634.h5',
        mono_energy_keV=7.246961723347276,
    )

    metadata = reader.read(master_path).get_metadata()

    assert metadata.probe_energy_eV == pytest.approx(7246.961723347276)
    assert metadata.detector_distance_m is None
    assert metadata.beam_center is None


def test_missing_metadata_loads_with_none(
    reader: PolarDiffractionFileReader, tmp_path: Path
) -> None:
    """A missing optional key must not fail the load; the settings value stands."""
    write_eiger(tmp_path / 'eiger' / 'scan_000001.h5', num_patterns=3)
    master_path = write_master(
        tmp_path / 'scan_000001_master.hdf', eiger_filename='eiger/scan_000001.h5'
    )

    dataset = reader.read(master_path)
    metadata = dataset.get_metadata()

    assert metadata.detector_distance_m is None
    assert metadata.beam_center is None
    assert metadata.probe_energy_eV is None
    assert metadata.num_patterns_per_array == [3]


def test_unique_id_gaps_are_preserved(reader: PolarDiffractionFileReader, tmp_path: Path) -> None:
    """Dropped Eiger frames leave a gap; the index array has to carry it."""
    write_eiger(tmp_path / 'eiger' / 'scan_000002.h5', num_patterns=4, uid=[7, 8, 11, 12])
    master_path = write_master(
        tmp_path / 'scan_000002_master.hdf', eiger_filename='eiger/scan_000002.h5'
    )

    dataset = reader.read(master_path)

    assert list(dataset[0].get_indexes()) == [1, 2, 5, 6]


def test_fly_scan_patterns_pair_with_triggers_offset_by_one(tmp_path: Path) -> None:
    """Fly scan: skip-one. Pattern index ``k`` pairs with trigger ``k``; trigger 0 is spare.

    The diffraction reader emits ``uid - uid[0] + 1``, 1-based, while the softGlueZynq
    trigger counter is raw and 0-based, so one trigger is recorded before the first
    frame. This is the offset ``process_flyscan.plot_data`` applies as ``x[1:]``.
    """
    num_triggers = 6
    # A distinct coordinate per trigger, so a misalignment shifts the values.
    x_nm = [1000.0 + trigger for trigger in range(num_triggers)]

    write_eiger(tmp_path / 'eiger' / 'scan_000450.h5', num_patterns=num_triggers - 1)
    write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000450.h5',
        trigger_indexes=range(num_triggers),
        x_nm=x_nm,
        y_nm=[0.0] * num_triggers,
    )
    master_path = write_master(
        tmp_path / 'scan_000450_master.hdf',
        plan_name='flyscan',
        eiger_filename='eiger/scan_000450.h5',
        pos_stream_filename='pos_stream/scan_000450.h5',
    )

    pattern_indexes = list(PolarDiffractionFileReader().read(master_path)[0].get_indexes())
    positions = PolarPositionFileReader().read(master_path)
    position_indexes = [point.index for point in positions]

    assert pattern_indexes == [1, 2, 3, 4, 5]
    assert position_indexes == [0, 1, 2, 3, 4, 5]
    # Every pattern has an exact position; position 0 is the one left over.
    assert pattern_indexes == position_indexes[1:]

    by_index = {point.index: point for point in positions}

    # Negated, because the readers report probe-relative coordinates rather than the
    # stage readback the stream holds.
    for pattern_index in pattern_indexes:
        assert by_index[pattern_index].x_m == pytest.approx(-x_nm[pattern_index] * NANOMETER_M)


def test_step_scan_patterns_pair_with_positions_one_to_one(tmp_path: Path) -> None:
    """Step scan: strict 1:1. Both sides index off the same UID, so the ``+1`` cancels."""
    num_patterns = 5
    motor_x = [10.0 + index for index in range(num_patterns)]

    write_eiger(tmp_path / 'eiger' / 'scan_001634.h5', num_patterns=num_patterns)
    master_path = write_master(
        tmp_path / 'scan_001634_master.hdf',
        plan_name='rel_grid_scan',
        eiger_filename='eiger/scan_001634.h5',
        primary={
            'huber_hp_nanox': motor_x,
            'huber_hp_nanoy': [20.0] * num_patterns,
        },
        motors_yaml='- huber_hp_nanox\n- huber_hp_nanoy\n',
    )

    pattern_indexes = list(PolarDiffractionFileReader().read(master_path)[0].get_indexes())
    positions = PolarPositionFileReader().read(master_path)
    position_indexes = [point.index for point in positions]

    assert pattern_indexes == [1, 2, 3, 4, 5]
    assert pattern_indexes == position_indexes

    by_index = {point.index: point for point in positions}

    for ordinal, pattern_index in enumerate(pattern_indexes):
        assert by_index[pattern_index].x_m == pytest.approx(-motor_x[ordinal] * 1e-6)


def test_gapped_fly_scan_pairing_survives_dropped_frames(tmp_path: Path) -> None:
    """A dropped frame must not slide the remaining patterns onto the wrong triggers."""
    num_triggers = 6
    x_nm = [1000.0 + trigger for trigger in range(num_triggers)]

    # Frames 1, 2, 4, 5 recorded; frame 3 dropped.
    write_eiger(tmp_path / 'eiger' / 'scan_000451.h5', num_patterns=4, uid=[41, 42, 44, 45])
    write_pos_stream(
        tmp_path / 'pos_stream' / 'scan_000451.h5',
        trigger_indexes=range(num_triggers),
        x_nm=x_nm,
        y_nm=[0.0] * num_triggers,
    )
    master_path = write_master(
        tmp_path / 'scan_000451_master.hdf',
        plan_name='flyscan',
        eiger_filename='eiger/scan_000451.h5',
        pos_stream_filename='pos_stream/scan_000451.h5',
    )

    pattern_indexes = list(PolarDiffractionFileReader().read(master_path)[0].get_indexes())
    by_index = {point.index: point for point in PolarPositionFileReader().read(master_path)}

    assert pattern_indexes == [1, 2, 4, 5]

    # Negated, because the readers report probe-relative coordinates rather than the
    # stage readback the stream holds.
    for pattern_index in pattern_indexes:
        assert by_index[pattern_index].x_m == pytest.approx(-x_nm[pattern_index] * NANOMETER_M)


def test_polar_diffraction_reader_registers() -> None:
    registry = PluginRegistry.load_plugins()
    names = {plugin.simple_name for plugin in registry.diffraction_file_readers}

    assert 'APS_Polar' in names
