"""Readers for NSLS-II 3-ID Hard X-ray Nanoprobe files.

The fixtures are shrunk copies of the real layouts: Style 2 square-root amplitudes stored
FFT-shifted with their scalar metadata, Style 1 raw frames with a ``/scan`` group and an
I0 map, and the MATLAB export.
"""

from __future__ import annotations

from pathlib import Path
import logging

import h5py
import numpy
import pytest

from ptychodus.api.constants import wavelength_m_to_energy_eV
from ptychodus.api.diffraction import CropRegion
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.nsls2_hxn_file import (
    NSLS2HXNDiffractionFileReader,
    NSLS2HXNPositionFileReader,
    NSLS2MATLABDiffractionFileReader,
    NSLS2MATLABProductFileReader,
)

MICROMETER_M = 1e-6


def _counts(num_frames: int = 7, size: int = 6) -> numpy.ndarray:
    rng = numpy.random.default_rng(0)
    return rng.integers(0, 50, size=(num_frames, size, size)).astype(numpy.uint32)


def _write_style2(path: Path, counts: numpy.ndarray, *, ic_length: int | None = None) -> Path:
    num_frames = counts.shape[0]
    amplitudes = numpy.fft.ifftshift(numpy.sqrt(counts.astype(numpy.float64)), axes=(-2, -1))

    with h5py.File(path, 'w') as h5_file:
        h5_file['diffamp'] = amplitudes
        h5_file['points'] = numpy.stack(
            [numpy.arange(num_frames, dtype=float), 10.0 + numpy.arange(num_frames)]
        )
        h5_file['ic'] = 1000.0 + numpy.arange(num_frames if ic_length is None else ic_length)
        h5_file['lambda_nm'] = 0.10331673138943305
        h5_file['z_m'] = 2.05
        h5_file['ccd_pixel_um'] = 75.0
        h5_file['angle'] = -88.0

    return path


def _write_style1(
    path: Path,
    counts: numpy.ndarray,
    *,
    num1: int,
    num2: int,
    raw_frames: bool = False,
    motor: str = 'zpssx',
    unique_ids: numpy.ndarray | None = None,
) -> Path:
    num_frames = counts.shape[0]

    with h5py.File(path, 'w') as h5_file:
        if raw_frames:
            h5_file['/entry/data/data'] = counts
        else:
            h5_file['/diff_data/merlin1/det_images'] = counts

        if unique_ids is not None:
            h5_file['/entry/instrument/NDAttributes/NDArrayUniqueId'] = unique_ids

        h5_file['/diff_data/merlin1/Io'] = numpy.arange(num2 * num1, dtype=float).reshape(
            num2, num1
        )
        h5_file['/scan/scan_positions'] = numpy.stack(
            [numpy.arange(num_frames, dtype=float), -numpy.arange(num_frames, dtype=float)]
        )
        h5_file['/scan/energy'] = 8.0
        h5_file['/scan/exposure_time'] = 0.03
        h5_file['/scan/zp_theta'] = 76.94
        h5_file['/scan/mll_theta'] = -0.002
        h5_file['/scan/motors'] = numpy.array([motor.encode(), b'other'], dtype=object)

    return path


def test_style2_frames_are_squared_back_to_centered_counts(tmp_path: Path) -> None:
    counts = _counts()
    path = _write_style2(tmp_path / 'scan_250000.h5', counts)

    dataset = NSLS2HXNDiffractionFileReader(diffamp_block_size=3).read(path)
    metadata = dataset.get_metadata()

    assert metadata.pattern_dtype == numpy.dtype(numpy.uint32)
    assert metadata.num_patterns_per_array == [3, 3, 1]
    assert [array.get_indexes().tolist() for array in dataset] == [[0, 1, 2], [3, 4, 5], [6]]
    patterns = numpy.concatenate([array.get_patterns() for array in dataset])
    numpy.testing.assert_array_equal(patterns, counts)


def test_style2_cropping_follows_the_shift(tmp_path: Path) -> None:
    counts = _counts()
    path = _write_style2(tmp_path / 'scan_250000.h5', counts)
    region = CropRegion(x_range=(1, 4), y_range=(2, 5))

    patterns = NSLS2HXNDiffractionFileReader().read(path)[0].get_patterns(read_region=region)

    numpy.testing.assert_array_equal(patterns, counts[:, 2:5, 1:4])


def test_style2_amplitudes_that_are_not_counts_stay_float(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    counts = _counts().astype(numpy.float64) + 0.25
    path = _write_style2(tmp_path / 'scan_250000.h5', counts)  # type: ignore[arg-type]

    with caplog.at_level(logging.WARNING):
        dataset = NSLS2HXNDiffractionFileReader().read(path)

    assert dataset.get_metadata().pattern_dtype == numpy.dtype(numpy.float32)
    numpy.testing.assert_allclose(dataset[0].get_patterns(), counts, rtol=1e-6)
    assert 'not whole counts' in caplog.text


def test_style2_metadata(tmp_path: Path) -> None:
    path = _write_style2(tmp_path / 'scan_250000.h5', _counts())

    metadata = NSLS2HXNDiffractionFileReader().read(path).get_metadata()

    assert metadata.photon_energy_eV == pytest.approx(
        wavelength_m_to_energy_eV(0.10331673138943305e-9)
    )
    assert metadata.detector_distance_m == pytest.approx(2.05)
    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(75e-6)
    assert metadata.tomography_angle_deg == pytest.approx(-88.0)
    assert metadata.exposure_time_s is None


def test_style2_positions_carry_the_i0(tmp_path: Path) -> None:
    path = _write_style2(tmp_path / 'scan_250000.h5', _counts())

    positions = NSLS2HXNPositionFileReader().read(path)

    assert len(positions) == 7
    assert positions[2].x_m == pytest.approx(2.0 * MICROMETER_M)
    assert positions[2].y_m == pytest.approx(12.0 * MICROMETER_M)
    assert positions[2].probe_photon_count == pytest.approx(1002.0)


def test_a_mismatched_i0_is_dropped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_style2(tmp_path / 'scan_02250243.h5', _counts(), ic_length=21)

    with caplog.at_level(logging.WARNING):
        positions = NSLS2HXNPositionFileReader().read(path)

    assert all(point.probe_photon_count is None for point in positions)
    assert '21 I0 values for 7 positions' in caplog.text


@pytest.mark.parametrize('raw_frames', [False, True])
def test_style1_frames_and_metadata(tmp_path: Path, raw_frames: bool) -> None:
    counts = _counts(num_frames=6)
    path = _write_style1(
        tmp_path / 'scan_276751_merlin1.h5', counts, num1=3, num2=2, raw_frames=raw_frames
    )

    dataset = NSLS2HXNDiffractionFileReader().read(path)
    metadata = dataset.get_metadata()

    numpy.testing.assert_array_equal(dataset[0].get_patterns(), counts)
    assert metadata.photon_energy_eV == pytest.approx(8000.0)
    assert metadata.exposure_time_s == pytest.approx(0.03)
    assert metadata.detector_distance_m is None
    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(55e-6)
    assert metadata.tomography_angle_deg == pytest.approx(76.94)


def test_style1_angle_follows_the_scanning_microscope(tmp_path: Path) -> None:
    path = _write_style1(tmp_path / 'scan_1_merlin1.h5', _counts(4), num1=2, num2=2, motor='dssx')

    metadata = NSLS2HXNDiffractionFileReader().read(path).get_metadata()

    assert metadata.tomography_angle_deg == pytest.approx(-0.002)


def test_style1_positions_take_the_i0_map_in_scan_order(tmp_path: Path) -> None:
    path = _write_style1(tmp_path / 'scan_276751_merlin1.h5', _counts(6), num1=3, num2=2)

    positions = NSLS2HXNPositionFileReader().read(path)

    assert [point.probe_photon_count for point in positions] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert positions[4].x_m == pytest.approx(4.0 * MICROMETER_M)
    assert positions[4].y_m == pytest.approx(-4.0 * MICROMETER_M)


def test_style1_dropped_frames_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = _write_style1(
        tmp_path / 'scan_276585_merlin1.h5',
        _counts(4),
        num1=2,
        num2=2,
        raw_frames=True,
        unique_ids=numpy.array([1, 2, 4, 5]),
    )

    with caplog.at_level(logging.WARNING):
        NSLS2HXNDiffractionFileReader().read(path)

    assert 'not consecutive' in caplog.text


def test_a_file_without_hxn_datasets_names_what_was_tried(tmp_path: Path) -> None:
    path = tmp_path / 'other.h5'

    with h5py.File(path, 'w') as h5_file:
        h5_file['unrelated'] = numpy.zeros(3)

    with pytest.raises(ValueError, match='/diffamp'):
        NSLS2HXNDiffractionFileReader().read(path)

    with pytest.raises(ProbePositionParseError, match='/scan/scan_positions'):
        NSLS2HXNPositionFileReader().read(path)


def _write_matlab(path: Path) -> Path:
    with h5py.File(path, 'w') as h5_file:
        h5_file['det_data'] = _counts(4).astype(numpy.float64)
        h5_file['det_pixel_size'] = numpy.array([[55.0]])
        h5_file['det_dist'] = numpy.array([[500_000.0]])
        h5_file['energy'] = numpy.array([[12.0]])
        h5_file['th'] = numpy.array([[3.0]])
        mask = numpy.ones((6, 6))
        mask[2, 3] = 0.0
        h5_file['mask'] = mask
        h5_file['img_pixel_size_x'] = numpy.array([[0.005]])
        h5_file['img_pixel_size_y'] = numpy.array([[0.005]])
        h5_file['pos_xy'] = numpy.stack([numpy.arange(4.0), numpy.arange(4.0), -numpy.arange(4.0)])
        h5_file['prb'] = numpy.ones((6, 6))
        h5_file['obj'] = numpy.ones((10, 10))

    return path


def test_matlab_diffraction_metadata_and_mask(tmp_path: Path) -> None:
    dataset = NSLS2MATLABDiffractionFileReader().read(_write_matlab(tmp_path / 'data_L.mat'))
    metadata = dataset.get_metadata()

    assert metadata.photon_energy_eV == pytest.approx(12_000.0)
    assert metadata.detector_distance_m == pytest.approx(0.5)
    assert metadata.detector_pixel_geometry is not None
    assert metadata.detector_pixel_geometry.width_m == pytest.approx(55e-6)
    assert metadata.tomography_angle_deg == pytest.approx(3.0)
    # Stored transposed, as every 2-D array of a v7.3 file is.
    assert dataset.get_bad_pixels()[3, 2]
    assert int(dataset.get_bad_pixels().sum()) == 1


def test_matlab_frames_are_transposed_back(tmp_path: Path) -> None:
    path = _write_matlab(tmp_path / 'data_L.mat')
    region = CropRegion(x_range=(1, 4), y_range=(0, 2))

    with h5py.File(path) as h5_file:
        stored = h5_file['det_data'][()]

    array = NSLS2MATLABDiffractionFileReader().read(path)[0]

    numpy.testing.assert_array_equal(array.get_patterns(), numpy.swapaxes(stored, -1, -2))
    numpy.testing.assert_array_equal(
        array.get_patterns(read_region=region), numpy.swapaxes(stored, -1, -2)[:, 0:2, 1:4]
    )


def test_matlab_product_pixel_size_is_in_meters(tmp_path: Path) -> None:
    product = NSLS2MATLABProductFileReader().read(_write_matlab(tmp_path / 'data_L.mat'))

    assert product.object_.get_pixel_geometry().width_m == pytest.approx(5e-9)
    assert product.metadata.tomography_angle_deg == pytest.approx(3.0)
    assert product.probe_positions[1].y_m == pytest.approx(-1.0 * MICROMETER_M)


def test_registration() -> None:
    registry = PluginRegistry.load_plugins()
    hxn = {
        name: {plugin.simple_name for plugin in catalog if plugin.simple_name.startswith('NSLS')}
        for name, catalog in (
            ('diffraction', registry.diffraction_file_readers),
            ('positions', registry.probe_position_file_readers),
            ('products', registry.product_file_readers),
            ('probes', registry.probe_file_readers),
            ('objects', registry.object_file_readers),
        )
    }

    assert hxn == {
        'diffraction': {'NSLS_II_HXN', 'NSLS_II_MATLAB'},
        'positions': {'NSLS_II_HXN', 'NSLS_II_MATLAB'},
        'products': {'NSLS_II_MATLAB'},
        'probes': {'NSLS_II_MATLAB'},
        'objects': {'NSLS_II_MATLAB'},
    }
