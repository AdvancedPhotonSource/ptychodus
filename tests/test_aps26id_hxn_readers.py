"""Readers for CNM/APS 26-ID-C Hard X-ray Nanoprobe diffraction files.

The fixtures are shrunk copies of the three real layouts: a Medipix TIFF series, plain or
gzipped, and an Eiger HDF5 stack beside an instrument snapshot, each with an EPICS MDA
scan record wherever that layout keeps it.
"""

from __future__ import annotations

from pathlib import Path
import gzip
import logging

import h5py
import numpy
import pytest
import tifffile

from ptychodus.api.plugins import PluginRegistry
from ptychodus.plugins.aps26id_hxn_file import (
    APS26IDH5DiffractionFileReader,
    APS26IDTiffDiffractionFileReader,
)

from mda_file_fixtures import write_channel_mda

MDA_ENERGY_EV = 11_200.0
MDA_DISTANCE_MM = 890.0


def _write_mda(path: Path, *, num_points: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return write_channel_mda(
        path,
        positioner=[float(i) for i in range(num_points)],
        channels=[
            ('DCM Energy', [MDA_ENERGY_EV] * num_points),
            ('Det R', [MDA_DISTANCE_MM] * num_points),
            ('ZP Piezo X', [float(i) for i in range(num_points)]),
        ],
        channel_names=['26idbDCM:sm8.RBV', '26idcSOFT:sm5.RBV', '26idcnpi:m34.RBV'],
    )


def _write_tiff_series(
    directory: Path,
    frames: numpy.ndarray,
    *,
    scan: int = 29,
    first_frame: int = 5604,
    gzipped: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for offset, frame in enumerate(frames):
        path = directory / f'scan_{scan}_img_{first_frame + offset:06d}.tif'
        tifffile.imwrite(path, frame)

        if gzipped:
            gz_path = path.with_name(path.name + '.gz')
            gz_path.write_bytes(gzip.compress(path.read_bytes()))
            path.unlink()
            path = gz_path

        paths.append(path)

    return paths[0]


def _frames(num_frames: int = 3, dtype: type = numpy.uint16) -> numpy.ndarray:
    return numpy.arange(num_frames * 8 * 8, dtype=dtype).reshape(num_frames, 8, 8)


def test_tiff_series_is_numbered_in_frame_order(tmp_path: Path) -> None:
    frames = _frames()
    first = _write_tiff_series(tmp_path, frames)
    # Another scan in the same directory must not join the series.
    _write_tiff_series(tmp_path, _frames(2), scan=30, first_frame=9000)
    _write_mda(tmp_path / '26idbSOFT_0029.mda')

    dataset = APS26IDTiffDiffractionFileReader().read(first.with_name('scan_29_img_005605.tif'))

    assert len(dataset) == 3
    assert [int(array.get_indexes()[0]) for array in dataset] == [0, 1, 2]
    assert [array.get_label() for array in dataset] == [
        f'scan_29_img_{n:06d}' for n in (5604, 5605, 5606)
    ]
    numpy.testing.assert_array_equal(dataset[2].get_patterns()[0], frames[2])


def test_gzipped_tiff_series_reads(tmp_path: Path) -> None:
    frames = _frames(dtype=numpy.int32)
    first = _write_tiff_series(tmp_path / 'Images' / '126', frames, scan=126, gzipped=True)
    _write_mda(tmp_path / 'mda' / '26idbSOFT_0126.mda')

    dataset = APS26IDTiffDiffractionFileReader().read(first)
    metadata = dataset.get_metadata()

    assert first.name.endswith('.tif.gz')
    assert sum(metadata.num_patterns_per_array) == 3
    assert str(metadata.file_path).endswith('scan_126_img_(\\d+).tif.gz')
    numpy.testing.assert_array_equal(dataset[1].get_patterns()[0], frames[1])


def test_signed_frames_are_read_as_unsigned_counts(tmp_path: Path) -> None:
    """Counts above 32767 wrap negative in int16; they are restored, not discarded."""
    counts = numpy.full((1, 8, 8), 40_000, dtype=numpy.uint16)
    first = _write_tiff_series(tmp_path, counts.view(numpy.int16))

    dataset = APS26IDTiffDiffractionFileReader().read(first)
    patterns = dataset[0].get_patterns()

    assert dataset.get_metadata().pattern_dtype == numpy.dtype(numpy.uint16)
    assert patterns.dtype == numpy.uint16
    assert int(patterns.min()) == 40_000


def test_32_bit_frames_keep_their_sign(tmp_path: Path) -> None:
    """An int32 -1 flags an invalid reading for the value filter; it must not become 2**32 - 1."""
    frames = _frames(dtype=numpy.int32)
    frames[0, 2, 2] = -1
    first = _write_tiff_series(tmp_path, frames)

    dataset = APS26IDTiffDiffractionFileReader().read(first)

    assert dataset.get_metadata().pattern_dtype == numpy.dtype(numpy.int32)
    assert int(dataset[0].get_patterns()[0, 2, 2]) == -1


def test_seam_pixels_are_bad_when_asked(tmp_path: Path) -> None:
    first = _write_tiff_series(tmp_path, _frames())

    bad_pixels = APS26IDTiffDiffractionFileReader(seam_pixels=(3, 4)).read(first).get_bad_pixels()

    assert bad_pixels[3, :].all() and bad_pixels[:, 4].all()
    assert int(bad_pixels.sum()) == 8 * 8 - 6 * 6
    # Masking the seam is opt-in.
    assert not APS26IDTiffDiffractionFileReader().read(first).get_bad_pixels().any()


@pytest.mark.parametrize(
    'mda_location',
    [
        Path('26idbSOFT_0029.mda'),
        Path('..') / '26idbSOFT_0029.mda',
        Path('..') / 'mda' / '26idbSOFT_0029.mda',
        Path('..') / '..' / 'mda' / '26idbSOFT_0029.mda',
    ],
)
def test_mda_metadata_is_found_where_each_layout_keeps_it(
    tmp_path: Path, mda_location: Path
) -> None:
    frame_directory = tmp_path / 'a' / 'b'
    first = _write_tiff_series(frame_directory, _frames())
    _write_mda((frame_directory / mda_location).resolve())

    metadata = APS26IDTiffDiffractionFileReader().read(first).get_metadata()

    assert metadata.photon_energy_eV == pytest.approx(MDA_ENERGY_EV)
    assert metadata.detector_distance_m == pytest.approx(MDA_DISTANCE_MM * 1e-3)
    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(55e-6)


def test_a_missing_mda_leaves_the_metadata_unset(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    first = _write_tiff_series(tmp_path, _frames())

    with caplog.at_level(logging.WARNING):
        metadata = APS26IDTiffDiffractionFileReader().read(first).get_metadata()

    assert metadata.photon_energy_eV is None
    assert metadata.detector_distance_m is None
    assert '26idbSOFT_0029.mda' in caplog.text


def _write_eiger_scan(
    directory: Path,
    frames: numpy.ndarray,
    *,
    snapshot_energy_eV: float | None = 10_000.0,  # noqa: N803
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / 'scan_43_000000.h5'
    data = directory / 'scan_43_000099.h5'

    with h5py.File(snapshot, 'w') as h5_file:
        group = h5_file.create_group('/entry/instrument/26-ID-C')
        group['count_time'] = 0.3
        h5_file.create_group('/entry/instrument/26-ID-B')

        if snapshot_energy_eV is not None:
            h5_file['/entry/instrument/26-ID-B/Channel Cut Energy'] = snapshot_energy_eV
            h5_file['/entry/instrument/26-ID-B/DCM Energy'] = snapshot_energy_eV - 85.0

    with h5py.File(data, 'w') as h5_file:
        h5_file['/entry/data/data'] = frames

    return snapshot, data


def test_eiger_scan_opens_from_either_member(tmp_path: Path) -> None:
    frames = _frames(dtype=numpy.int32)
    snapshot, data = _write_eiger_scan(tmp_path / 'h5', frames)
    _write_mda(tmp_path / 'mda' / '26idbSOFT_0043.mda')

    for path in (snapshot, data):
        dataset = APS26IDH5DiffractionFileReader().read(path)
        metadata = dataset.get_metadata()

        assert metadata.num_patterns_per_array == [3]
        assert dataset[0].get_indexes().tolist() == [0, 1, 2]
        assert metadata.exposure_time_s == pytest.approx(0.3)
        assert metadata.detector_pixel_geometry is not None
        assert metadata.detector_pixel_geometry.width_m == pytest.approx(75e-6)


def test_snapshot_values_override_the_mda(tmp_path: Path) -> None:
    """The snapshot's Channel Cut Energy wins; Det R, which it lacks, comes from the MDA."""
    snapshot, _ = _write_eiger_scan(tmp_path / 'h5', _frames(dtype=numpy.int32))
    _write_mda(tmp_path / 'mda' / '26idbSOFT_0043.mda')

    metadata = APS26IDH5DiffractionFileReader().read(snapshot).get_metadata()

    assert metadata.photon_energy_eV == pytest.approx(10_000.0)
    assert metadata.detector_distance_m == pytest.approx(MDA_DISTANCE_MM * 1e-3)


def test_eiger_flags_in_the_first_frame_are_bad_pixels(tmp_path: Path) -> None:
    """A flag only in a later frame is left to the value filter, not masked."""
    frames = numpy.ones((3, 8, 8), dtype=numpy.int32)
    frames[:, 1, 2] = -1
    frames[2, 5, 5] = -1
    snapshot, _ = _write_eiger_scan(tmp_path, frames)

    bad_pixels = APS26IDH5DiffractionFileReader().read(snapshot).get_bad_pixels()

    assert bad_pixels[1, 2]
    assert not bad_pixels[5, 5]
    assert int(bad_pixels.sum()) == 1


def test_frame_index_offset_shifts_every_pattern(tmp_path: Path) -> None:
    snapshot, _ = _write_eiger_scan(tmp_path, _frames(dtype=numpy.int32))

    dataset = APS26IDH5DiffractionFileReader(frame_index_offset=1).read(snapshot)

    assert dataset[0].get_indexes().tolist() == [1, 2, 3]


def test_a_scan_with_no_frames_is_refused(tmp_path: Path) -> None:
    snapshot, data = _write_eiger_scan(tmp_path, _frames(dtype=numpy.int32))
    data.unlink()

    with pytest.raises(ValueError, match='/entry/data/data'):
        APS26IDH5DiffractionFileReader().read(snapshot)


def test_both_readers_register() -> None:
    registry = PluginRegistry.load_plugins()

    assert isinstance(
        registry.diffraction_file_readers.get_strategy_by_name('CNM_APS_HXN_TIFF'),
        APS26IDTiffDiffractionFileReader,
    )
    assert isinstance(
        registry.diffraction_file_readers.get_strategy_by_name('CNM_APS_HXN'),
        APS26IDH5DiffractionFileReader,
    )
