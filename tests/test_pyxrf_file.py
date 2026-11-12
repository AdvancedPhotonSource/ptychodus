"""Tests for the PyXRF fluorescence reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy
import numpy.testing
import pytest

from ptychodus.plugins.nsls2_pyxrf_file import PyXRFFileReader

_NAMES = ('Fe_K', 'Pt_L', 'compton', 'snip_bkg', 'r_factor', 'sel_cnt', 'total_cnt')


def _write_pyxrf_file(
    file_path: Path,
    *,
    names: tuple[str, ...] = _NAMES,
    dwell_s: float | None = 0.01,
    name_dtype: str = 'S20',
) -> numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.float64]]:
    rng = numpy.random.default_rng(0)
    maps = rng.uniform(0.0, 100.0, size=(len(names), 3, 4))

    with h5py.File(file_path, 'w') as h5_file:
        detsum = h5_file.require_group('xrfmap/detsum')
        detsum.create_dataset('xrf_fit', data=maps)

        if name_dtype == 'str':
            detsum.create_dataset('xrf_fit_name', data=list(names), dtype=h5py.string_dtype())
        else:
            detsum.create_dataset('xrf_fit_name', data=list(names), dtype=name_dtype)

        if dwell_s is not None:
            scan_metadata = h5_file.require_group('xrfmap/scan_metadata')
            scan_metadata.attrs['file_format'] = 'NSLS2-XRF-MAP'
            scan_metadata.attrs['param_dwell'] = dwell_s

    return maps


def test_diagnostic_channels_are_skipped(tmp_path: Path) -> None:
    file_path = tmp_path / 'scan2D_1.h5'
    _write_pyxrf_file(file_path)

    dataset = PyXRFFileReader().read(file_path)

    assert [element_map.name for element_map in dataset] == ['Fe_K', 'Pt_L', 'compton']


def test_maps_are_divided_by_the_dwell_time(tmp_path: Path) -> None:
    file_path = tmp_path / 'scan2D_1.h5'
    maps = _write_pyxrf_file(file_path, dwell_s=0.01)

    dataset = PyXRFFileReader().read(file_path)

    for index, element_map in enumerate(dataset):
        numpy.testing.assert_allclose(element_map.counts_per_second, maps[index] / 0.01)


def test_a_file_without_scan_metadata_reads_unscaled(tmp_path: Path) -> None:
    file_path = tmp_path / 'scan2D_1.h5'
    maps = _write_pyxrf_file(file_path, dwell_s=None)

    dataset = PyXRFFileReader().read(file_path)

    numpy.testing.assert_allclose(dataset[0].counts_per_second, maps[0])


def test_variable_length_string_names_decode(tmp_path: Path) -> None:
    file_path = tmp_path / 'scan2D_1.h5'
    _write_pyxrf_file(file_path, names=('Ca_K', 'elastic'), name_dtype='str')

    dataset = PyXRFFileReader().read(file_path)

    assert [element_map.name for element_map in dataset] == ['Ca_K', 'elastic']


def test_dataset_records_the_pyxrf_paths(tmp_path: Path) -> None:
    file_path = tmp_path / 'scan2D_1.h5'
    _write_pyxrf_file(file_path)

    dataset = PyXRFFileReader().read(file_path)

    assert dataset.counts_per_second_path == '/xrfmap/detsum/xrf_fit'
    assert dataset.channel_names_path == '/xrfmap/detsum/xrf_fit_name'


def test_a_file_without_fitted_maps_raises(tmp_path: Path) -> None:
    file_path = tmp_path / 'raw_only.h5'

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('xrfmap/detsum/counts', data=numpy.zeros((3, 4, 8)))

    with pytest.raises(ValueError, match='xrf_fit'):
        PyXRFFileReader().read(file_path)
