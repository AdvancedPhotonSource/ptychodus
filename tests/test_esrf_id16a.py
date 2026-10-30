"""Unit tests for the ESRF ID16A near-field ptychography readers.

The geometry assertions use the real 55 nm scan's numbers, so they pin the
conversion against values measured on the instrument rather than against whatever
the code happens to compute.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.api.propagate import compute_magnification
from ptychodus.plugins.esrf_id16a import (
    ESRFID16ADarkBadPixelsFileReader,
    ESRFID16ADarkFileReader,
    ESRFID16ANearFieldDiffractionFileReader,
    ESRFID16ANearFieldPositionFileReader,
)

from esrf_file_fixtures import (
    DARK_LEVEL,
    FRAMES_PER_FILE,
    OPTIC_USED_UM,
    PIXEL_SIZE_UM,
    PROPAGATION_DISTANCE_MM,
    write_lima_file,
    write_scan,
)

_MICROMETER = 1.0e-6
_MILLIMETER = 1.0e-3


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    return write_scan(tmp_path / 'scan')


def test_geometry_round_trips_the_recorded_propagation_distance(scan: Path) -> None:
    metadata = ESRFID16ANearFieldDiffractionFileReader().read(scan).get_metadata()

    assert metadata.detector_distance_m is not None
    assert metadata.focus_object_distance_m is not None
    magnification = compute_magnification(
        metadata.detector_distance_m, metadata.focus_object_distance_m
    )

    # The header states the equivalent parallel-beam distance; ptychodus derives it
    # back out of the lab-frame pair. The two conventions must agree exactly.
    assert metadata.detector_distance_m / magnification == pytest.approx(
        PROPAGATION_DISTANCE_MM * _MILLIMETER
    )


def test_object_pixel_matches_the_recorded_pixel_size(scan: Path) -> None:
    metadata = ESRFID16ANearFieldDiffractionFileReader().read(scan).get_metadata()

    assert metadata.detector_distance_m is not None
    assert metadata.focus_object_distance_m is not None
    assert metadata.detector_pixel_geometry is not None
    magnification = compute_magnification(
        metadata.detector_distance_m, metadata.focus_object_distance_m
    )

    assert metadata.detector_pixel_geometry.width_m / magnification == pytest.approx(
        PIXEL_SIZE_UM * _MICROMETER
    )
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(OPTIC_USED_UM * _MICROMETER)


def test_scan_is_declared_near_field_with_an_upstream_focus(scan: Path) -> None:
    metadata = ESRFID16ANearFieldDiffractionFileReader().read(scan).get_metadata()

    assert metadata.far_field is False
    assert metadata.focus_object_distance_m is not None
    # Negative places the focus upstream of the sample, the diverging-cone case.
    assert metadata.focus_object_distance_m < 0.0


def test_header_values_survive_their_padding(scan: Path) -> None:
    metadata = ESRFID16ANearFieldDiffractionFileReader().read(scan).get_metadata()

    # Both the trailing-space key ('energy ') and the padded value (' 17.1') have to
    # be stripped for this to parse at all.
    assert metadata.photon_energy_eV == pytest.approx(17100.0)
    assert metadata.exposure_time_s == pytest.approx(0.2)


def test_master_and_frame_file_agree(scan: Path) -> None:
    reader = ESRFID16ANearFieldDiffractionFileReader()
    frame_file = scan.parent / 'sample_055nm_nfp010000.h5'

    from_master = reader.read(scan).get_metadata()
    from_frame = reader.read(frame_file).get_metadata()

    for field in (
        'detector_distance_m',
        'focus_object_distance_m',
        'far_field',
        'photon_energy_eV',
        'detector_pixel_geometry',
        'detector_extent',
    ):
        assert getattr(from_master, field) == getattr(from_frame, field)


def test_detector_group_is_found_under_another_name(tmp_path: Path) -> None:
    # The group name is a detector-interface label, so the reader probes for the one
    # holding `data` rather than trusting the name the real files happen to use.
    directory = tmp_path / 'scan'
    master = write_scan(directory)
    write_lima_file(
        directory / 'sample_055nm_nfp010000.h5',
        spy=[0.0] * FRAMES_PER_FILE,
        spz=[0.0] * FRAMES_PER_FILE,
        detector_name='Ximea',
    )

    metadata = ESRFID16ANearFieldDiffractionFileReader().read(master).get_metadata()

    assert metadata.detector_extent.width_px == 8
    assert metadata.far_field is False


def test_indexes_run_consecutively_across_a_series(tmp_path: Path) -> None:
    master = write_scan(tmp_path / 'scan', num_files=2)
    dataset = ESRFID16ANearFieldDiffractionFileReader().read(master)

    indexes = numpy.concatenate([array.get_indexes() for array in dataset])
    assert indexes.tolist() == list(range(2 * FRAMES_PER_FILE))


def test_positions_pair_with_patterns_by_index(tmp_path: Path) -> None:
    master = write_scan(tmp_path / 'scan', num_files=2)
    dataset = ESRFID16ANearFieldDiffractionFileReader().read(master)
    positions = ESRFID16ANearFieldPositionFileReader().read(master)

    pattern_indexes = numpy.concatenate([array.get_indexes() for array in dataset])
    assert [point.index for point in positions] == pattern_indexes.tolist()


def test_a_reversed_repeat_reverses_the_positions(tmp_path: Path) -> None:
    # The real nfp02 repeat traverses the same points backwards, so array order and
    # scan order disagree between the pair while every index still finds its pattern.
    forward = write_scan(tmp_path / 'nfp01', num_files=2)
    backward = write_scan(tmp_path / 'nfp02', num_files=2, reverse=True)

    reader = ESRFID16ANearFieldPositionFileReader()
    x_forward = [point.x_m for point in reader.read(forward)]
    x_backward = [point.x_m for point in reader.read(backward)]

    assert x_backward == list(reversed(x_forward))
    assert [point.index for point in reader.read(backward)] == list(range(len(x_backward)))


def test_positions_are_micrometers_unscaled(scan: Path) -> None:
    positions = ESRFID16ANearFieldPositionFileReader().read(scan)

    # First fixture point is spy = -5.0 um, spz = +5.0 um, emitted without sign flips.
    assert positions[0].x_m == pytest.approx(-5.0 * _MICROMETER)
    assert positions[0].y_m == pytest.approx(+5.0 * _MICROMETER)


def test_a_position_list_shorter_than_the_frames_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / 'scan'
    master = write_scan(directory)
    write_lima_file(
        directory / 'sample_055nm_nfp010000.h5',
        spy=[0.0] * FRAMES_PER_FILE,
        spz=[0.0] * (FRAMES_PER_FILE - 1),
    )

    with pytest.raises(ProbePositionParseError):
        ESRFID16ANearFieldPositionFileReader().read(master)


def test_dark_reader_returns_the_per_pixel_mean(scan: Path) -> None:
    background = ESRFID16ADarkFileReader().read(scan.parent / 'dark_0000.h5')

    assert background.ndim == 2
    assert background.dtype == numpy.float64
    # Away from the seeded hot and stuck pixels the stack is a flat pedestal.
    assert background[4, 4] == pytest.approx(DARK_LEVEL)


def test_bad_pixels_flags_hot_and_saturated(scan: Path) -> None:
    mask = ESRFID16ADarkBadPixelsFileReader().read(scan.parent / 'dark_0000.h5')

    assert mask.shape == (8, 8)
    assert mask.dtype == numpy.bool_
    assert mask[1, 2]  # hot: twenty times the median dark level
    assert mask[0, 0]  # saturated in one frame
    assert not mask[4, 4]


def test_hot_pixel_factor_above_the_outlier_spares_it(scan: Path) -> None:
    mask = ESRFID16ADarkBadPixelsFileReader(hot_pixel_factor=50.0).read(
        scan.parent / 'dark_0000.h5'
    )

    assert not mask[1, 2]
    assert mask[0, 0]  # saturation does not depend on the factor


def test_readers_are_registered() -> None:
    registry = PluginRegistry.load_plugins()

    assert registry.diffraction_file_readers.get_strategy_by_name('ESRF_ID16A_NFP') is not None
    assert registry.probe_position_file_readers.get_strategy_by_name('ESRF_ID16A_NFP') is not None
    assert registry.background_file_readers.get_strategy_by_name('ESRF_ID16A_Dark') is not None
    assert registry.bad_pixels_file_readers.get_strategy_by_name('ESRF_ID16A_Dark') is not None
