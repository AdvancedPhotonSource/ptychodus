"""Unit tests for the NPZ archive PtychoPINN trains from.

Nothing here imports ptycho_torch: the writer is pure numpy, so the format is tested
on a bare install whether or not the backend is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import h5py
import numpy
import numpy.testing

from ptychodus.api.io import load_cxi_product, save_cxi_reconstruct_input
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import zero_bad_pixels
from ptychodus.plugins.ptychopinn.training_file import (
    PtychoPINNTrainingDataFileWriter,
    save_ptychopinn_training_data,
)

from test_io import _make_reconstruct_input

_PTYCHOPINN_NPZ_KEYS: Final[set[str]] = {
    'xcoords',
    'ycoords',
    'xcoords_start',
    'ycoords_start',
    'diff3d',
    'bad_pixels',
    'probeGuess',
    'objectGuess',
    'scan_index',
    'scale_contract_version',
    'measurement_domain',
}


class TestSavePtychopinnTrainingData:
    def test_writes_the_expected_keys(self, tmp_path: Path) -> None:
        """The eight original keys survive, plus the bad-pixel mask."""
        path = tmp_path / 'training.npz'
        save_ptychopinn_training_data(path, _make_reconstruct_input())

        with numpy.load(path) as npz_file:
            assert set(npz_file.files) == _PTYCHOPINN_NPZ_KEYS

    def test_writes_every_probe_mode(self, tmp_path: Path) -> None:
        """probeGuess is always the full mode stack, never just mode zero."""
        parameters = _make_reconstruct_input(num_incoherent_modes=4, with_opr=True)
        path = tmp_path / 'training.npz'
        save_ptychopinn_training_data(path, parameters)

        expected = parameters.product.probes.get_array()[0]

        with numpy.load(path) as npz_file:
            assert npz_file['probeGuess'].shape == (4, 8, 8)
            numpy.testing.assert_array_equal(npz_file['probeGuess'], expected)

    def test_bad_pixels_are_inpainted_and_recorded(self, tmp_path: Path) -> None:
        """diff3d is repaired and the mask says which positions were filled."""
        parameters = _make_reconstruct_input()
        assert parameters.bad_pixels is not None
        bad = parameters.bad_pixels
        path = tmp_path / 'training.npz'
        save_ptychopinn_training_data(path, parameters)

        with numpy.load(path) as npz_file:
            numpy.testing.assert_array_equal(npz_file['bad_pixels'], bad)
            assert numpy.all(npz_file['diff3d'][:, bad] != 0)
            assert npz_file['diff3d'].dtype == parameters.diffraction_patterns.dtype

    def test_inpaint_disabled_matches_zero_bad_pixels(self, tmp_path: Path) -> None:
        """The toggle off reproduces the previous behaviour exactly."""
        parameters = _make_reconstruct_input()
        path = tmp_path / 'training.npz'
        save_ptychopinn_training_data(path, parameters, inpaint_bad_pixels=False)

        expected = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)

        with numpy.load(path) as npz_file:
            numpy.testing.assert_array_equal(npz_file['diff3d'], expected)

    def test_scan_index_is_still_all_zeros(self, tmp_path: Path) -> None:
        """The single-object assumption is unchanged."""
        parameters = _make_reconstruct_input()
        path = tmp_path / 'training.npz'
        save_ptychopinn_training_data(path, parameters)

        with numpy.load(path) as npz_file:
            assert not npz_file['scan_index'].any()
            assert len(npz_file['scan_index']) == len(parameters.product.probe_positions)

    def test_agrees_with_the_cxi_writer(self, tmp_path: Path) -> None:
        """Both formats carry the same quantities for the same input.

        The two writers share no code past the repair step, so nothing but this
        keeps them from drifting apart on what an exported input contains.
        """
        parameters = _make_reconstruct_input(num_incoherent_modes=3)
        npz_path = tmp_path / 'training.npz'
        cxi_path = tmp_path / 'training.cxi'
        save_ptychopinn_training_data(npz_path, parameters)
        save_cxi_reconstruct_input(cxi_path, parameters, inpaint_bad_pixels=True)
        loaded = load_cxi_product(cxi_path)
        geometry = loaded.object_.get_geometry()

        with numpy.load(npz_path) as npz_file, h5py.File(cxi_path, 'r') as h5_file:
            entry = h5_file['entry_1']
            numpy.testing.assert_array_equal(npz_file['diff3d'], entry['data_1/data'][()])
            numpy.testing.assert_array_equal(
                npz_file['bad_pixels'], entry['instrument_1/detector_1/mask'][()] != 0
            )
            numpy.testing.assert_array_equal(npz_file['probeGuess'], entry['image_2/data'][()])
            numpy.testing.assert_array_equal(npz_file['objectGuess'], entry['image_1/data'][()])
            numpy.testing.assert_allclose(
                npz_file['xcoords'],
                geometry.map_probe_positions_to_object_x_px(loaded.probe_positions),
            )
            numpy.testing.assert_allclose(
                npz_file['ycoords'],
                geometry.map_probe_positions_to_object_y_px(loaded.probe_positions),
            )

    def test_declares_patterns_as_photon_counts(self, tmp_path: Path) -> None:
        """The archive says which scaling it is in, rather than leaving it guessed.

        A reader compares this pair against the scaling its model was trained
        under and converts or rejects accordingly. An undeclared archive is
        instead interpreted from context, so the declaration is what keeps an
        already-counted export from being rescaled as amplitude.
        """
        parameters = _make_reconstruct_input()
        path = tmp_path / 'training.npz'
        save_ptychopinn_training_data(path, parameters)

        with numpy.load(path) as npz_file:
            assert str(npz_file['scale_contract_version']) == 'ci_intensity_v2'
            assert str(npz_file['measurement_domain']) == 'count_intensity'


class TestPtychoPINNTrainingDataFileWriter:
    def test_writes_the_archive_at_the_given_path(self, tmp_path: Path) -> None:
        path = tmp_path / 'training.npz'

        PtychoPINNTrainingDataFileWriter().write(path, _make_reconstruct_input())

        with numpy.load(path) as npz_file:
            assert set(npz_file.files) == _PTYCHOPINN_NPZ_KEYS

    def test_inpainting_can_be_turned_off(self, tmp_path: Path) -> None:
        parameters = _make_reconstruct_input()
        path = tmp_path / 'training.npz'

        PtychoPINNTrainingDataFileWriter(inpaint_bad_pixels=False).write(path, parameters)

        expected = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)

        with numpy.load(path) as npz_file:
            numpy.testing.assert_array_equal(npz_file['diff3d'], expected)

    def test_the_writer_is_registered_under_the_library_name(self) -> None:
        # The shared name is what makes this the format offered by default whenever
        # PtychoPINN is the selected backend.
        registry = PluginRegistry.load_plugins()

        assert registry.reconstructor_libraries.find_plugin('PtychoPINN')
        writer = registry.training_data_file_writers.get_strategy_by_name('PtychoPINN')
        assert isinstance(writer, PtychoPINNTrainingDataFileWriter)
