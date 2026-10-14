"""Tests for how the diffraction readers identify the members of one scan's series.

Fixtures are synthetic: they reproduce the file-naming conventions observed at each
instrument rather than any one acquisition. The naming is the point -- every reader here
globs siblings out of a directory, and picking the wrong digit field silently assembles
frames from unrelated scans instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy
import pytest
import tifffile

from ptychodus.api.diffraction import CropRegion
from ptychodus.plugins.aps02id_diffraction_file import APS2IDDiffractionFileReader
from ptychodus.plugins.mda.mda_position_file import RASTER_LINE_INDEX_STRIDE
from ptychodus.plugins.aps12id_diffraction_file import APS12IDDiffractionFileReader
from ptychodus.plugins.aps19id_isn_diffraction_file import ISNDiffractionFileReader
from ptychodus.plugins.h5_diffraction_file import H5DiffractionFileReader
from ptychodus.plugins.tiff_diffraction_file import TiffDiffractionFileReader

_DETECTOR_H = 4
_DETECTOR_W = 6
_DATA_PATH = '/entry/data/data'


def _write_h5(path: Path, num_frames: int, *, data_path: str = _DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = (num_frames, _DETECTOR_H, _DETECTOR_W)
    data = numpy.arange(numpy.prod(shape), dtype=numpy.uint32).reshape(shape)

    with h5py.File(path, 'w') as h5_file:
        h5_file.create_dataset(data_path, data=data)

    return path


def _totals(reader, path: Path) -> tuple[int, list[int], list[str]]:
    dataset = reader.read(path)
    counts = list(dataset.get_metadata().num_patterns_per_array)
    labels = [array.get_label() for array in dataset]
    return len(counts), counts, labels


def test_2id_series_is_scoped_to_one_scan(tmp_path: Path) -> None:
    """2-ID-E names the scan and the frame with the same width, so length cannot choose.

    `fly054_data_001.h5` has two three-digit fields. Selecting the longest digit run ties
    and resolves toward the scan number, which globs frame 1 of every scan in the
    directory -- the defect this pins.
    """
    for scan in (53, 54, 55):
        for frame in (1, 2, 3):
            _write_h5(tmp_path / f'fly{scan:03d}_data_{frame:03d}.h5', 2)

    num_arrays, counts, labels = _totals(
        APS2IDDiffractionFileReader(), tmp_path / 'fly054_data_001.h5'
    )

    assert num_arrays == 3
    assert counts == [2, 2, 2]
    assert labels == ['fly054_data_001', 'fly054_data_002', 'fly054_data_003']


def test_2id_series_handles_a_six_digit_frame_field(tmp_path: Path) -> None:
    """The Bionanoprobe's frame field is wider than its scan field, and must still win."""
    for scan in (674, 675):
        for frame in range(3):
            _write_h5(tmp_path / f'bnp_fly{scan:04d}_{frame:06d}.h5', 2)

    num_arrays, _, labels = _totals(
        APS2IDDiffractionFileReader(), tmp_path / 'bnp_fly0675_000000.h5'
    )

    assert num_arrays == 3
    assert all(label.startswith('bnp_fly0675_') for label in labels)


def test_2id_reader_reads_each_files_own_frame_count(tmp_path: Path) -> None:
    """A cut-short line must survive rather than be dropped during assembly.

    Declaring the first file's count for every array makes any array that disagrees fail
    its length check, and the failure surfaces only as a warning.
    """
    for frame, num_frames in enumerate((3, 5, 2), start=1):
        _write_h5(tmp_path / f'fly054_data_{frame:03d}.h5', num_frames)

    dataset = APS2IDDiffractionFileReader().read(tmp_path / 'fly054_data_001.h5')

    assert list(dataset.get_metadata().num_patterns_per_array) == [3, 5, 2]

    # Indexes run continuously across the ragged arrays, so the patterns still pair with
    # their positions by index.
    indexes = numpy.concatenate([array.get_indexes() for array in dataset])
    assert indexes.tolist() == list(range(10))


def test_2id_reader_supplies_the_eiger_pitch(tmp_path: Path) -> None:
    _write_h5(tmp_path / 'fly054_data_001.h5', 2)
    metadata = APS2IDDiffractionFileReader().read(tmp_path / 'fly054_data_001.h5').get_metadata()

    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(75e-6)


def test_isn_series_is_scoped_to_one_scan(tmp_path: Path) -> None:
    """The older ISN naming puts a four-digit scan ahead of a three-digit frame."""
    for scan in (179, 180):
        for frame in (0, 1, 2):
            _write_h5(tmp_path / f'19ide_{scan:04d}_{frame:03d}.h5', 2)

    num_arrays, _, labels = _totals(ISNDiffractionFileReader(), tmp_path / '19ide_0179_000.h5')

    assert num_arrays == 3
    assert all(label.startswith('19ide_0179_') for label in labels)


def test_isn_reader_supplies_a_detector_pitch(tmp_path: Path) -> None:
    """ISN files record no geometry, and a dataset with no pitch cannot build a probe."""
    _write_h5(tmp_path / '19ide_0179_000.h5', 2)
    metadata = ISNDiffractionFileReader().read(tmp_path / '19ide_0179_000.h5').get_metadata()

    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(75e-6)


def test_12id_series_collects_every_line_and_point(tmp_path: Path) -> None:
    """12-ID writes one file per point, so the series is keyed by (line, point).

    Matching a single digit field picks the line and pins the point, collecting one point
    from each line and discarding the rest.
    """
    shape = (_DETECTOR_H, _DETECTOR_W)

    for line in (1, 2):
        for point in (0, 1, 2):
            path = tmp_path / f'007_{line:05d}_{point:d}.h5'
            with h5py.File(path, 'w') as h5_file:
                h5_file.create_dataset(_DATA_PATH, data=numpy.zeros(shape, dtype=numpy.uint32))

    dataset = APS12IDDiffractionFileReader().read(tmp_path / '007_00001_0.h5')
    labels = [array.get_label() for array in dataset]

    assert labels == [
        '007_00001_0',
        '007_00001_1',
        '007_00001_2',
        '007_00002_0',
        '007_00002_1',
        '007_00002_2',
    ]


def test_12id_single_frame_files_assemble(tmp_path: Path) -> None:
    """A one-frame-per-file layout stores a 2-D dataset, but an array must yield a stack.

    Returned bare, the pattern fails the 3-D check in AssembledDiffractionData and the
    array is dropped with only a warning -- every array, so the scan assembles to
    nothing. Cropping is worse: the region slice indexes an axis that is not there.
    """
    shape = (_DETECTOR_H, _DETECTOR_W)

    for point in (0, 1):
        path = tmp_path / f'007_00001_{point:d}.h5'
        with h5py.File(path, 'w') as h5_file:
            h5_file.create_dataset(_DATA_PATH, data=numpy.ones(shape, dtype=numpy.uint32))

    dataset = APS12IDDiffractionFileReader().read(tmp_path / '007_00001_0.h5')

    for array in dataset:
        patterns = array.get_patterns()
        assert patterns.shape == (1, _DETECTOR_H, _DETECTOR_W)

    cropped = dataset[0].get_patterns(read_region=CropRegion(x_range=(1, 3), y_range=(1, 3)))
    assert cropped.shape == (1, 2, 2)


def test_tiff_series_is_scoped_and_counts_its_own_pages(tmp_path: Path) -> None:
    """The TIFF reader shares both defects: the naming tie and the assumed page count."""
    for scan in (54, 55):
        for frame, num_pages in enumerate((3, 1, 2), start=1):
            data = numpy.zeros((num_pages, _DETECTOR_H, _DETECTOR_W), dtype=numpy.uint16)
            # minisblack keeps each frame a separate page; the default would store a
            # three-frame stack as one RGB page with three component planes.
            tifffile.imwrite(
                tmp_path / f'fly{scan:03d}_data_{frame:03d}.tif', data, photometric='minisblack'
            )

    dataset = TiffDiffractionFileReader().read(tmp_path / 'fly054_data_001.tif')
    labels = [array.get_label() for array in dataset]

    assert labels == ['fly054_data_001', 'fly054_data_002', 'fly054_data_003']
    assert list(dataset.get_metadata().num_patterns_per_array) == [3, 1, 2]

    # Each array carries one index per page it actually holds; a lone index on a stacked
    # file would not match what get_patterns returns.
    indexes = numpy.concatenate([array.get_indexes() for array in dataset])
    assert indexes.tolist() == list(range(6))


def test_tiff_series_spans_an_unpadded_counter_rollover(tmp_path: Path) -> None:
    """areaDetector pads the counter to three digits and then lets it grow.

    A scan past 999 frames therefore holds both `_999` and `_1000`. Requiring the counter
    width of the member that was opened splits that series in two, and the reader loads
    whichever half it was pointed at without saying it dropped the other -- 999 frames of
    1005, silently paired against 1005 positions.
    """
    data = numpy.zeros((_DETECTOR_H, _DETECTOR_W), dtype=numpy.uint16)

    for scan in (15, 16):
        for frame in range(1, 1006):
            tifffile.imwrite(
                tmp_path / f'Star_scan{scan}_9p5keV_{frame:03d}.tif', data, photometric='minisblack'
            )

    reader = TiffDiffractionFileReader()

    # The entry point must not matter: three- and four-digit members name one series.
    for member in ('Star_scan15_9p5keV_001.tif', 'Star_scan15_9p5keV_1005.tif'):
        dataset = reader.read(tmp_path / member)
        labels = [array.get_label() for array in dataset]

        assert len(labels) == 1005
        assert labels[0] == 'Star_scan15_9p5keV_001'
        assert labels[-1] == 'Star_scan15_9p5keV_1005'

        # Frames sort numerically, not lexicographically: 1000 follows 999, not 001.
        assert labels[998:1001] == [
            'Star_scan15_9p5keV_999',
            'Star_scan15_9p5keV_1000',
            'Star_scan15_9p5keV_1001',
        ]

        # Widening the counter must not widen the scope; scan 16 stays out.
        assert not any('scan16' in label for label in labels)


def test_h5_reader_supplies_a_pitch_only_when_given_one(tmp_path: Path) -> None:
    """The fold_slice registration passes a pitch; the bare reader still reports none."""
    path = _write_h5(tmp_path / 'data_roi0_Ndp4_dp.hdf5', 3, data_path='/dp')

    without = H5DiffractionFileReader(data_path='/dp').read(path).get_metadata()
    assert without.detector_pixel_geometry is None

    with_pitch = (
        H5DiffractionFileReader(data_path='/dp', detector_pixel_size_m=75e-6)
        .read(path)
        .get_metadata()
    )
    assert with_pitch.detector_pixel_geometry is not None
    assert with_pitch.detector_pixel_geometry.height_m == pytest.approx(75e-6)


# --- 2-ID-E line-major numbering
#
# The pairing these produce, against a real MDA raster, is covered in
# tests/test_mda_position_file.py; these pin what the diffraction side contributes.


def _indexes(reader, path: Path) -> list[int]:
    return [int(i) for array in reader.read(path) for i in array.get_indexes()]


def test_2id_reader_numbers_by_running_count_by_default(tmp_path: Path) -> None:
    """2-ID-D and the Bionanoprobe keep the contiguous numbering, so the stride is opt-in."""
    for frame in (1, 2, 3):
        _write_h5(tmp_path / f'fly007_data_{frame:03d}.h5', 2)

    reader = APS2IDDiffractionFileReader()

    assert _indexes(reader, tmp_path / 'fly007_data_001.h5') == [0, 1, 2, 3, 4, 5]


def test_2id_reader_numbers_line_major_when_given_a_stride(tmp_path: Path) -> None:
    """A pattern is named by its line and column, so the line comes off the file counter."""
    for frame in (1, 2, 3):
        _write_h5(tmp_path / f'fly007_data_{frame:03d}.h5', 2)

    reader = APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE)

    assert _indexes(reader, tmp_path / 'fly007_data_001.h5') == [
        0,
        1,
        RASTER_LINE_INDEX_STRIDE,
        RASTER_LINE_INDEX_STRIDE + 1,
        2 * RASTER_LINE_INDEX_STRIDE,
        2 * RASTER_LINE_INDEX_STRIDE + 1,
    ]


def test_2id_reader_keeps_line_numbers_across_a_missing_member(tmp_path: Path) -> None:
    """Reading the line off the counter is what keeps a hole from shifting later lines."""
    for frame in (1, 3):
        _write_h5(tmp_path / f'fly007_data_{frame:03d}.h5', 2)

    reader = APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE)

    assert _indexes(reader, tmp_path / 'fly007_data_001.h5') == [
        0,
        1,
        2 * RASTER_LINE_INDEX_STRIDE,
        2 * RASTER_LINE_INDEX_STRIDE + 1,
    ]


def test_2id_reader_rejects_a_line_that_reaches_the_stride(tmp_path: Path) -> None:
    """A longer line would run into the next line's numbers, so it is refused."""
    _write_h5(tmp_path / 'fly007_data_001.h5', 4)

    reader = APS2IDDiffractionFileReader(line_index_stride=3)

    with pytest.raises(ValueError, match='reaches the line-major'):
        reader.read(tmp_path / 'fly007_data_001.h5')


def test_2id_reader_rejects_a_series_numbered_from_zero(tmp_path: Path) -> None:
    """Line zero is member one; a zero counter would make the first line negative."""
    for frame in (0, 1):
        _write_h5(tmp_path / f'fly007_data_{frame:03d}.h5', 2)

    reader = APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE)

    with pytest.raises(ValueError, match='one-based counter'):
        reader.read(tmp_path / 'fly007_data_000.h5')


def test_2id_reader_skips_an_unreadable_member(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An interrupted acquisition leaves a zero-length member; the rest is still a scan."""
    for frame in (1, 2, 3):
        _write_h5(tmp_path / f'fly007_data_{frame:03d}.h5', 2)

    (tmp_path / 'fly007_data_002.h5').write_bytes(b'')

    reader = APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE)

    with caplog.at_level('WARNING'):
        indexes = _indexes(reader, tmp_path / 'fly007_data_001.h5')

    # The survivors keep their own line numbers, so line two is still line two.
    assert indexes == [0, 1, 2 * RASTER_LINE_INDEX_STRIDE, 2 * RASTER_LINE_INDEX_STRIDE + 1]
    assert 'Skipping unreadable' in caplog.text


def test_2id_reader_rejects_a_series_with_nothing_readable(tmp_path: Path) -> None:
    (tmp_path / 'fly007_data_001.h5').write_bytes(b'')

    reader = APS2IDDiffractionFileReader()

    with pytest.raises(ValueError, match='No readable diffraction files'):
        reader.read(tmp_path / 'fly007_data_001.h5')


def test_2id_reader_widens_the_dtype_across_the_series(tmp_path: Path) -> None:
    """A narrow first member must not size a buffer the rest of the series overflows."""
    for frame, dtype in ((1, numpy.uint16), (2, numpy.uint32)):
        path = tmp_path / f'fly007_data_{frame:03d}.h5'

        with h5py.File(path, 'w') as h5_file:
            h5_file.create_dataset(
                _DATA_PATH, data=numpy.zeros((2, _DETECTOR_H, _DETECTOR_W), dtype=dtype)
            )

    metadata = APS2IDDiffractionFileReader().read(tmp_path / 'fly007_data_001.h5').get_metadata()

    assert metadata.pattern_dtype == numpy.dtype(numpy.uint32)


def test_2id_reader_rejects_a_member_of_a_different_size(tmp_path: Path) -> None:
    _write_h5(tmp_path / 'fly007_data_001.h5', 2)

    with h5py.File(tmp_path / 'fly007_data_002.h5', 'w') as h5_file:
        h5_file.create_dataset(_DATA_PATH, data=numpy.zeros((2, _DETECTOR_H + 1, _DETECTOR_W)))

    with pytest.raises(ValueError, match='but the series is'):
        APS2IDDiffractionFileReader().read(tmp_path / 'fly007_data_001.h5')


def test_2id_reader_warns_when_the_detector_dropped_frames(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A frame dropped mid-line shifts every column after it, and only the counter says so."""
    path = _write_h5(tmp_path / 'fly007_data_001.h5', 4)

    with h5py.File(path, 'a') as h5_file:
        h5_file.create_dataset(
            '/entry/instrument/NDAttributes/NDArrayUniqueId',
            data=numpy.array([10, 11, 13, 14], dtype=numpy.int32),
        )

    reader = APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE)

    with caplog.at_level('WARNING'):
        reader.read(path)

    assert 'dropped frames' in caplog.text


def test_2id_reader_is_quiet_when_the_frame_counter_is_consecutive(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_h5(tmp_path / 'fly007_data_001.h5', 4)

    with h5py.File(path, 'a') as h5_file:
        h5_file.create_dataset(
            '/entry/instrument/NDAttributes/NDArrayUniqueId',
            data=numpy.array([10, 11, 12, 13], dtype=numpy.int32),
        )

    reader = APS2IDDiffractionFileReader(line_index_stride=RASTER_LINE_INDEX_STRIDE)

    with caplog.at_level('WARNING'):
        reader.read(path)

    assert caplog.text == ''
