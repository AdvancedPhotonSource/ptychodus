"""Tests for the APS 12-ID-E readers -- the 2026 master-file layout, processed HDF5, TIFF
and valid-pixel mask -- and the pre-APS-U Bionanoprobe position reader.

Fixtures are synthetic. The 2026 fixtures reproduce the layout observed in 12-ID-E data
from 2026_2 on; the processed, TIFF and Bionanoprobe fixtures pin the readers against
the documented layouts, as no such data was available when they were written. Treat a
failure here as a reader regression and a surprise from real data as a fixture that
needs correcting.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy
import pytest
import tifffile

from ptychodus.plugins.aps02id_bnp_position_file import BionanoprobeMAPSPositionFileReader
from ptychodus.plugins.aps12id_file import APS12IDDiffractionFileReader, APS12IDPositionFileReader
from ptychodus.plugins.aps12id_processed_file import (
    APS12IDProcessedDiffractionFileReader,
    APS12IDProcessedPositionFileReader,
)
from ptychodus.plugins.aps12id_tiff_file import (
    APS12IDTIFFDiffractionFileReader,
    APS12IDTIFFPositionFileReader,
)
from ptychodus.plugins.bad_pixels import APS12IDValidPixelMaskFileReader
from ptychodus.api.probe_positions import ProbePositionParseError

_NM = 1e-9
_UM = 1e-6

_DETECTOR_H = 6
_DETECTOR_W = 5
_POINTS_PER_LINE = 3
_NUM_LINES = 2


_MM = 1e-3


def _write_master(
    scan_directory: Path,
    *,
    positions_mm: numpy.ndarray | None,
    exposure: bool = True,
    sample_name: str | None = 'siemens_star',
) -> Path:
    """A 2026 master file holding the geometry and, when given, the nominal positions."""
    scan_directory.mkdir(parents=True, exist_ok=True)
    master = scan_directory / 'sample_0042_master.h5'

    with h5py.File(master, 'w') as h5_file:
        h5_file['/entry/instrument/monochromator/monoE'] = 8.0
        h5_file['/entry/instrument/detector/distance'] = 10000.0
        h5_file['/entry/instrument/detector/x_pixel_size'] = 0.172
        h5_file['/entry/instrument/detector/y_pixel_size'] = 0.175
        h5_file['/entry/sample/theta'] = 15.0

        if exposure:
            h5_file['/entry/instrument/detector/ExposureTime'] = 0.002

        if sample_name is not None:
            h5_file['/entry/sample/sample_name'] = sample_name

        if positions_mm is not None:
            h5_file['/entry/sample/positions'] = positions_mm

    return master


def _frame(value: int) -> numpy.ndarray:
    """One detector frame with a module-gap (-1) pixel and a dead (-2) pixel."""
    frame = numpy.full((_DETECTOR_H, _DETECTOR_W), value, dtype=numpy.int32)
    frame[0, 0] = -1
    frame[1, 2] = -2
    return frame


def _write_step_scan(directory: Path, num_points: int = 3) -> Path:
    """A step scan: a master plus one file per point, each holding a single 2-D frame."""
    scan_directory = directory / 'ptycho' / 'S0042'
    positions_mm = numpy.array([[0.001 * point, -0.002 * point] for point in range(num_points)])
    master = _write_master(scan_directory, positions_mm=positions_mm)

    for point in range(num_points):
        with h5py.File(scan_directory / f'sample_0042_{point + 1:05d}.h5', 'w') as h5_file:
            h5_file['/entry/data/data'] = _frame(point)

    return master


class TestAPS12IDReaders:
    def test_step_scan_reads_one_array_per_frame_file(self, tmp_path: Path) -> None:
        master = _write_step_scan(tmp_path)

        dataset = APS12IDDiffractionFileReader().read(master)
        indexes = numpy.concatenate([array.get_indexes() for array in dataset])

        assert dataset.get_metadata().num_patterns_per_array == [1, 1, 1]
        numpy.testing.assert_array_equal(indexes, numpy.arange(3))
        assert [int(array.get_patterns()[0, -1, -1]) for array in dataset] == [0, 1, 2]

    def test_fly_scan_reads_a_stack_from_one_file(self, tmp_path: Path) -> None:
        scan_directory = tmp_path / 'S0042'
        master = _write_master(scan_directory, positions_mm=None)

        with h5py.File(scan_directory / 'sample_0042_00001.h5', 'w') as h5_file:
            h5_file['/entry/data/data'] = numpy.stack([_frame(value) for value in range(4)])

        dataset = APS12IDDiffractionFileReader().read(master)

        assert dataset.get_metadata().num_patterns_per_array == [4]
        numpy.testing.assert_array_equal(dataset[0].get_indexes(), numpy.arange(4))

    def test_metadata_comes_from_the_master_in_ptychodus_units(self, tmp_path: Path) -> None:
        metadata = APS12IDDiffractionFileReader().read(_write_step_scan(tmp_path)).get_metadata()

        assert metadata.photon_energy_eV == pytest.approx(8000.0)
        assert metadata.exposure_time_s == pytest.approx(0.002)
        assert metadata.detector_distance_m == pytest.approx(10.0)
        assert metadata.detector_pixel_geometry is not None
        assert metadata.detector_pixel_geometry.width_m == pytest.approx(172e-6)
        assert metadata.detector_pixel_geometry.height_m == pytest.approx(175e-6)
        assert metadata.tomography_angle_deg == pytest.approx(15.0)
        assert metadata.sample_name == 'siemens_star'
        assert metadata.detector_extent.width_px == _DETECTOR_W
        assert metadata.detector_extent.height_px == _DETECTOR_H

    def test_an_unrecorded_field_is_left_unset(self, tmp_path: Path) -> None:
        """A master predating a field must read as None, not fail."""
        scan_directory = tmp_path / 'S0042'
        master = _write_master(scan_directory, positions_mm=None, exposure=False, sample_name=None)

        with h5py.File(scan_directory / 'sample_0042_00001.h5', 'w') as h5_file:
            h5_file['/entry/data/data'] = _frame(0)

        metadata = APS12IDDiffractionFileReader().read(master).get_metadata()

        assert metadata.exposure_time_s is None
        assert metadata.sample_name is None

    def test_negative_pixels_are_bad(self, tmp_path: Path) -> None:
        bad_pixels = (
            APS12IDDiffractionFileReader().read(_write_step_scan(tmp_path)).get_bad_pixels()
        )

        expected = numpy.zeros((_DETECTOR_H, _DETECTOR_W), dtype=bool)
        expected[0, 0] = True
        expected[1, 2] = True
        numpy.testing.assert_array_equal(bad_pixels, expected)

    def test_any_scan_file_is_an_entry_point(self, tmp_path: Path) -> None:
        master = _write_step_scan(tmp_path)
        raw_positions = tmp_path / 'positions' / 'S0042' / 'sample_0042_00001.h5'
        raw_positions.parent.mkdir(parents=True)
        raw_positions.touch()

        for entry in (master, master.with_name('sample_0042_00002.h5'), raw_positions):
            dataset = APS12IDDiffractionFileReader().read(entry)
            positions = APS12IDPositionFileReader().read(entry)

            assert dataset.get_metadata().file_path == master
            assert len(dataset) == 3
            assert len(positions) == 3

    def test_frame_files_of_another_scan_are_ignored(self, tmp_path: Path) -> None:
        master = _write_step_scan(tmp_path)

        with h5py.File(master.with_name('other_0042_00001.h5'), 'w') as h5_file:
            h5_file['/entry/data/data'] = _frame(9)

        assert len(APS12IDDiffractionFileReader().read(master)) == 3

    def test_an_unreadable_frame_file_ends_the_scan(self, tmp_path: Path) -> None:
        """Skipping it would pair every later frame with the wrong position."""
        master = _write_step_scan(tmp_path, num_points=4)
        master.with_name('sample_0042_00003.h5').write_bytes(b'not hdf5')

        dataset = APS12IDDiffractionFileReader().read(master)

        assert [array.get_label() for array in dataset] == [
            'sample_0042_00001',
            'sample_0042_00002',
        ]

    def test_a_scan_without_a_master_is_rejected(self, tmp_path: Path) -> None:
        with h5py.File(tmp_path / 'sample_042_00001.h5', 'w') as h5_file:
            h5_file['/entry/data/data'] = _frame(0)

        with pytest.raises(ValueError, match='No master file'):
            APS12IDDiffractionFileReader().read(tmp_path / 'sample_042_00001.h5')

        with pytest.raises(ProbePositionParseError, match='No master file'):
            APS12IDPositionFileReader().read(tmp_path / 'sample_042_00001.h5')

    def test_positions_scale_from_millimeters_in_traversal_order(self, tmp_path: Path) -> None:
        positions = APS12IDPositionFileReader().read(_write_step_scan(tmp_path))

        assert [point.index for point in positions] == [0, 1, 2]
        assert [point.x_m for point in positions] == pytest.approx([0.0, 1e-6, 2e-6])
        assert [point.y_m for point in positions] == pytest.approx([0.0, -2e-6, -4e-6])

    def test_a_master_without_positions_is_rejected(self, tmp_path: Path) -> None:
        master = _write_master(tmp_path / 'S0042', positions_mm=None)

        with pytest.raises(ProbePositionParseError, match='records no scan positions'):
            APS12IDPositionFileReader().read(master)

    def test_valid_pixel_mask_stored_as_uint8(self, tmp_path: Path) -> None:
        valid = numpy.ones((_DETECTOR_H, _DETECTOR_W), dtype=numpy.uint8)
        valid[2, 3] = 0
        mask_file = tmp_path / 'pixel_mask.h5'

        with h5py.File(mask_file, 'w') as h5_file:
            h5_file['valid_pixel_mask'] = valid

        bad_pixels = APS12IDValidPixelMaskFileReader().read(mask_file)

        numpy.testing.assert_array_equal(bad_pixels, valid == 0)


def _write_processed_scan(directory: Path, *, beam_center_yx: tuple[float, float] | None) -> Path:
    """A master file plus one file per line, each with /dp and /positions."""
    directory.mkdir(parents=True, exist_ok=True)
    master = directory / 'sample_007_master.h5'

    with h5py.File(master, 'w') as h5_file:
        if beam_center_yx is not None:
            h5_file.attrs['beam_center_YX'] = numpy.asarray(beam_center_yx)

    for line in range(_NUM_LINES):
        with h5py.File(directory / f'sample_007_{line:05d}.h5', 'w') as h5_file:
            patterns = numpy.arange(
                _POINTS_PER_LINE * _DETECTOR_H * _DETECTOR_W, dtype=numpy.uint32
            ).reshape(_POINTS_PER_LINE, _DETECTOR_H, _DETECTOR_W)
            h5_file.create_dataset('dp', data=patterns + line)

            # Columns: [unused, y_nm, x_nm]
            positions = numpy.array(
                [[0.0, 100.0 * line, 10.0 * point] for point in range(_POINTS_PER_LINE)]
            )
            h5_file.create_dataset('positions', data=positions)

    return master


class TestAPS12IDProcessedReaders:
    def test_reads_every_line_into_one_dataset(self, tmp_path: Path) -> None:
        master = _write_processed_scan(tmp_path, beam_center_yx=(3.0, 2.0))

        dataset = APS12IDProcessedDiffractionFileReader().read(master)

        assert len(dataset) == _NUM_LINES
        assert dataset.get_metadata().num_patterns_per_array == [_POINTS_PER_LINE] * _NUM_LINES

    def test_indexes_run_consecutively_across_lines(self, tmp_path: Path) -> None:
        """Per-line restarts would silently pair line two with line one's positions."""
        master = _write_processed_scan(tmp_path, beam_center_yx=None)

        dataset = APS12IDProcessedDiffractionFileReader().read(master)
        indexes = numpy.concatenate([array.get_indexes() for array in dataset])

        numpy.testing.assert_array_equal(indexes, numpy.arange(_NUM_LINES * _POINTS_PER_LINE))

    def test_reports_the_master_beam_center_as_xy(self, tmp_path: Path) -> None:
        """The attribute is stored (y, x); BeamCenter is (x, y)."""
        master = _write_processed_scan(tmp_path, beam_center_yx=(3.0, 2.0))

        beam_center = (
            APS12IDProcessedDiffractionFileReader().read(master).get_metadata().beam_center
        )

        assert beam_center is not None
        assert (beam_center.x_px, beam_center.y_px) == (2, 3)

    def test_reports_the_pilatus_pitch(self, tmp_path: Path) -> None:
        master = _write_processed_scan(tmp_path, beam_center_yx=None)

        geometry = APS12IDProcessedDiffractionFileReader().read(master).get_metadata()

        assert geometry.detector_pixel_geometry is not None
        assert geometry.detector_pixel_geometry.width_m == pytest.approx(172e-6)

    def test_a_line_file_is_an_equally_valid_entry_point(self, tmp_path: Path) -> None:
        _write_processed_scan(tmp_path, beam_center_yx=(3.0, 2.0))
        line_file = tmp_path / 'sample_007_00001.h5'

        dataset = APS12IDProcessedDiffractionFileReader().read(line_file)

        assert len(dataset) == _NUM_LINES

    def test_positions_negate_x_and_scale_from_nanometers(self, tmp_path: Path) -> None:
        master = _write_processed_scan(tmp_path, beam_center_yx=None)

        positions = APS12IDProcessedPositionFileReader().read(master)

        assert len(positions) == _NUM_LINES * _POINTS_PER_LINE
        assert positions[0].y_m == pytest.approx(0.0)
        assert positions[1].x_m == pytest.approx(-10.0 * _NM)
        assert positions[_POINTS_PER_LINE].y_m == pytest.approx(100.0 * _NM)

    def test_positions_and_patterns_agree_in_count(self, tmp_path: Path) -> None:
        master = _write_processed_scan(tmp_path, beam_center_yx=None)

        dataset = APS12IDProcessedDiffractionFileReader().read(master)
        positions = APS12IDProcessedPositionFileReader().read(master)

        assert sum(dataset.get_metadata().num_patterns_per_array) == len(positions)


def _write_tiff_scan(directory: Path, *, rows_per_point: int = 1) -> Path:
    """One TIFF per scan point beside one .dat per scan point."""
    directory.mkdir(parents=True, exist_ok=True)

    for line in range(_NUM_LINES):
        for point in range(_POINTS_PER_LINE):
            stem = f'sample_007_{line:05d}_{point}'
            frame = numpy.full((_DETECTOR_H, _DETECTOR_W), line * 10 + point, dtype=numpy.uint16)
            tifffile.imwrite(directory / f'{stem}.tif', frame)

            rows = numpy.array(
                [[0.0, 100.0 * line, 10.0 * point + jitter] for jitter in range(rows_per_point)]
            )
            numpy.savetxt(directory / f'{stem}.dat', rows)

    return directory / 'sample_007_00000_0.tif'


class TestAPS12IDTIFFReaders:
    def test_reads_every_frame_in_scan_order(self, tmp_path: Path) -> None:
        first = _write_tiff_scan(tmp_path)

        dataset = APS12IDTIFFDiffractionFileReader().read(first)
        values = [int(array.get_patterns()[0, 0, 0]) for array in dataset]

        assert values == [0, 1, 2, 10, 11, 12]

    def test_indexes_are_consecutive(self, tmp_path: Path) -> None:
        first = _write_tiff_scan(tmp_path)

        dataset = APS12IDTIFFDiffractionFileReader().read(first)
        indexes = numpy.concatenate([array.get_indexes() for array in dataset])

        numpy.testing.assert_array_equal(indexes, numpy.arange(_NUM_LINES * _POINTS_PER_LINE))

    def test_reports_extent_and_the_pilatus_pitch(self, tmp_path: Path) -> None:
        first = _write_tiff_scan(tmp_path)

        metadata = APS12IDTIFFDiffractionFileReader().read(first).get_metadata()

        assert metadata.detector_extent is not None
        assert metadata.detector_extent.width_px == _DETECTOR_W
        assert metadata.detector_extent.height_px == _DETECTOR_H
        assert metadata.detector_pixel_geometry is not None
        assert metadata.detector_pixel_geometry.width_m == pytest.approx(172e-6)

    def test_any_member_is_a_valid_entry_point(self, tmp_path: Path) -> None:
        _write_tiff_scan(tmp_path)

        dataset = APS12IDTIFFDiffractionFileReader().read(tmp_path / 'sample_007_00001_2.tif')

        assert len(dataset) == _NUM_LINES * _POINTS_PER_LINE

    def test_oversampled_rows_share_one_point_index(self, tmp_path: Path) -> None:
        """Duplicates are collapsed downstream, so they must carry the same index."""
        _write_tiff_scan(tmp_path, rows_per_point=4)

        positions = APS12IDTIFFPositionFileReader().read(tmp_path / 'sample_007_00000_0.dat')

        assert len(positions) == _NUM_LINES * _POINTS_PER_LINE * 4
        assert [point.index for point in positions][:4] == [0, 0, 0, 0]
        assert len({point.index for point in positions}) == _NUM_LINES * _POINTS_PER_LINE

    def test_positions_negate_x_and_scale_from_nanometers(self, tmp_path: Path) -> None:
        _write_tiff_scan(tmp_path)

        positions = APS12IDTIFFPositionFileReader().read(tmp_path / 'sample_007_00000_0.dat')

        assert positions[1].x_m == pytest.approx(-10.0 * _NM)
        assert positions[_POINTS_PER_LINE].y_m == pytest.approx(100.0 * _NM)

    def test_a_misnamed_file_is_rejected(self, tmp_path: Path) -> None:
        tifffile.imwrite(tmp_path / 'not-a-scan.tif', numpy.zeros((2, 2), dtype=numpy.uint16))

        with pytest.raises(ValueError):
            APS12IDTIFFDiffractionFileReader().read(tmp_path / 'not-a-scan.tif')


class TestBionanoprobeMAPSPositionReader:
    def _write(self, file: Path, x_um: list[float], y_um: list[float]) -> Path:
        with h5py.File(file, 'w') as h5_file:
            h5_file.create_dataset('MAPS/x_axis', data=numpy.asarray(x_um))
            h5_file.create_dataset('MAPS/y_axis', data=numpy.asarray(y_um))

        return file

    def test_expands_the_axis_vectors_into_a_row_major_grid(self, tmp_path: Path) -> None:
        file = self._write(tmp_path / 'bnp_fly0001.mda.h5', [0.0, 1.0, 2.0], [10.0, 20.0])

        positions = BionanoprobeMAPSPositionFileReader().read(file)

        assert len(positions) == 6
        assert [point.index for point in positions] == [0, 1, 2, 3, 4, 5]
        assert positions[0].x_m == pytest.approx(0.0)
        assert positions[1].x_m == pytest.approx(1.0 * _UM)
        assert positions[0].y_m == pytest.approx(10.0 * _UM)
        # The row advances only after every column of the previous one.
        assert positions[3].y_m == pytest.approx(20.0 * _UM)
        assert positions[3].x_m == pytest.approx(0.0)

    def test_positions_are_not_recentered(self, tmp_path: Path) -> None:
        """Reported as the file states them, matching the post-upgrade MDA reader."""
        file = self._write(tmp_path / 'bnp_fly0002.mda.h5', [5.0, 6.0], [7.0])

        positions = BionanoprobeMAPSPositionFileReader().read(file)

        assert positions[0].x_m == pytest.approx(5.0 * _UM)
        assert positions[0].y_m == pytest.approx(7.0 * _UM)

    def test_a_file_without_the_maps_group_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / 'empty.h5'

        with h5py.File(file, 'w'):
            pass

        with pytest.raises(ProbePositionParseError):
            BionanoprobeMAPSPositionFileReader().read(file)
