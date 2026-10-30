"""Unit tests for the PtychoFM training pair, written and read.

The format is two HDF5 files sharing a stem, associated by that suffix and nothing
else. Its parameter half is a subset of the ptychodus product layout, which is what
makes a round trip meaningful: what the exporter writes is what the reader has to
recover. The constraints pinned here are the trainer's -- a strictly four-dimensional
probe, a square pattern with a probe the same size, positions as long as the pattern
stack, and a span-based rule deciding whether coordinates are meters or pixels.

Nothing here imports ptycho_fm itself: it is published on no index, so a test that
needed it would not run on a bare install.
"""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy
import pytest

from ptychodus.api.io import ProductFileKeys, load_product, save_product
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.plugins.ptycho_fm.training_file import (
    PtychoFMProductFileReader,
    PtychoFMTrainingDataFileWriter,
    resolve_training_pair,
    save_training_data,
)

from test_io import _make_product

IMG_SIZE = 8
NUM_POSITIONS = 3
PIXEL_SIZE_M = 10e-9


def _make_input(**kwargs: object) -> ReconstructInput:
    defaults: dict[str, object] = {
        'num_positions': NUM_POSITIONS,
        'probe_height': IMG_SIZE,
        'probe_width': IMG_SIZE,
        'obj_height': 32,
        'obj_width': 32,
    }
    product = _make_product(**{**defaults, **kwargs})  # type: ignore[arg-type]
    rng = numpy.random.default_rng(7)
    patterns = rng.integers(0, 500, size=(NUM_POSITIONS, IMG_SIZE, IMG_SIZE)).astype(numpy.uint32)
    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=numpy.zeros((IMG_SIZE, IMG_SIZE), dtype=bool),
        product=product,
    )


class TestResolveTrainingPair:
    @pytest.mark.parametrize('name', ['scan_dp.hdf5', 'scan_para.hdf5', 'scan.hdf5'])
    def test_either_member_or_a_bare_stem_names_the_pair(self, tmp_path: Path, name: str) -> None:
        # The loader on the other side accepts either member, and the export dialog
        # supplies a stem that is neither.
        diffraction, parameter = resolve_training_pair(tmp_path / name)

        assert diffraction == tmp_path / 'scan_dp.hdf5'
        assert parameter == tmp_path / 'scan_para.hdf5'

    @pytest.mark.parametrize('name', ['scan_dp.h5', 'scan_para.h5', 'scan.h5'])
    def test_the_pair_suffixes_go_before_the_given_extension(
        self, tmp_path: Path, name: str
    ) -> None:
        diffraction, parameter = resolve_training_pair(tmp_path / name)

        assert diffraction == tmp_path / 'scan_dp.h5'
        assert parameter == tmp_path / 'scan_para.h5'

    @pytest.mark.parametrize('name', ['scan', 'scan.v2'])
    def test_a_name_without_an_hdf5_extension_is_given_one(self, tmp_path: Path, name: str) -> None:
        diffraction, parameter = resolve_training_pair(tmp_path / name)

        assert diffraction == tmp_path / f'{name}_dp.hdf5'
        assert parameter == tmp_path / f'{name}_para.hdf5'


class TestSaveTrainingData:
    def test_both_members_are_written_and_the_stem_is_not(self, tmp_path: Path) -> None:
        stem = tmp_path / 'scan.hdf5'

        save_training_data(stem, _make_input())

        assert (tmp_path / 'scan_dp.hdf5').is_file()
        assert (tmp_path / 'scan_para.hdf5').is_file()
        assert not stem.exists()

    def test_the_probe_is_four_dimensional(self, tmp_path: Path) -> None:
        # The loader rejects a three-dimensional probe outright, so the leading
        # OPR axis has to be present even when there is one of them.
        save_training_data(tmp_path / 'scan.hdf5', _make_input())

        with h5py.File(tmp_path / 'scan_para.hdf5', 'r') as h5_file:
            assert h5_file[ProductFileKeys.PROBE_ARRAY].ndim == 4

    def test_the_parameter_file_is_spelled_in_product_keys(self, tmp_path: Path) -> None:
        # The format is a subset of the product layout. Asserting the names keeps the
        # two descriptions of one thing from drifting apart.
        save_training_data(tmp_path / 'scan.hdf5', _make_input())

        with h5py.File(tmp_path / 'scan_para.hdf5', 'r') as h5_file:
            assert ProductFileKeys.OBJECT_ARRAY in h5_file
            assert ProductFileKeys.PROBE_ARRAY in h5_file
            assert ProductFileKeys.PROBE_POSITION_X in h5_file
            assert ProductFileKeys.PROBE_POSITION_Y in h5_file
            assert ProductFileKeys.PROBE_POSITION_INDEXES in h5_file
            attrs = h5_file[ProductFileKeys.OBJECT_ARRAY].attrs
            assert ProductFileKeys.OBJECT_PIXEL_HEIGHT in attrs

    def test_normalization_is_the_pattern_maximum(self, tmp_path: Path) -> None:
        parameters = _make_input()

        save_training_data(tmp_path / 'scan.hdf5', parameters)

        with h5py.File(tmp_path / 'scan_para.hdf5', 'r') as h5_file:
            assert h5_file['normalization'][()] == parameters.diffraction_patterns.max()

    def test_bad_pixels_are_zeroed(self, tmp_path: Path) -> None:
        parameters = _make_input()
        parameters.bad_pixels[0, 0] = True

        save_training_data(tmp_path / 'scan.hdf5', parameters)

        with h5py.File(tmp_path / 'scan_dp.hdf5', 'r') as h5_file:
            assert not h5_file['dp'][:, 0, 0].any()

    def test_patterns_that_are_not_square_are_refused(self, tmp_path: Path) -> None:
        # Nothing downstream resizes, so a mismatch surfaces as an opaque shape error
        # deep inside the model unless it is refused here.
        parameters = _make_input()
        parameters = replace(
            parameters,
            diffraction_patterns=parameters.diffraction_patterns[:, :, :-1],
            bad_pixels=parameters.bad_pixels[:, :-1],
        )

        with pytest.raises(ValueError, match='Diffraction patterns'):
            save_training_data(tmp_path / 'scan.hdf5', parameters)

    def test_a_probe_that_does_not_match_the_patterns_is_refused(self, tmp_path: Path) -> None:
        # The probe is stored at its native size beside the patterns, and is multiplied
        # into an object patch cut at the pattern size.
        parameters = _make_input(probe_height=IMG_SIZE + 2, probe_width=IMG_SIZE + 2)

        with pytest.raises(ValueError, match='Probe'):
            save_training_data(tmp_path / 'scan.hdf5', parameters)

    def test_nothing_is_written_when_refused(self, tmp_path: Path) -> None:
        parameters = _make_input(probe_height=IMG_SIZE + 2, probe_width=IMG_SIZE + 2)

        with pytest.raises(ValueError):
            save_training_data(tmp_path / 'scan.hdf5', parameters)

        assert not any(tmp_path.iterdir())


class TestPtychoFMTrainingDataFileWriter:
    def test_a_pair_written_as_h5_reads_back_through_both_readers(self, tmp_path: Path) -> None:
        parameters = _make_input()
        registry = PluginRegistry.load_plugins()

        PtychoFMTrainingDataFileWriter().write(tmp_path / 'scan.h5', parameters)

        assert sorted(path.name for path in tmp_path.iterdir()) == ['scan_dp.h5', 'scan_para.h5']
        diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(
            PtychoFMProductFileReader.SIMPLE_NAME
        )
        dataset = diffraction_reader.read(tmp_path / 'scan_dp.h5')
        assert numpy.array_equal(dataset[0].get_patterns(), parameters.diffraction_patterns)
        product = PtychoFMProductFileReader().read(tmp_path / 'scan_para.h5')
        assert len(product.probe_positions) == NUM_POSITIONS

    def test_the_readers_offer_the_extensions_the_writer_does(self) -> None:
        assert (
            PtychoFMProductFileReader().get_file_extensions()
            == PtychoFMTrainingDataFileWriter().get_file_extensions()
        )


class TestPtychoFMProductFileReader:
    def test_the_exported_pair_reads_back(self, tmp_path: Path) -> None:
        parameters = _make_input()
        save_training_data(tmp_path / 'scan.hdf5', parameters)

        product = PtychoFMProductFileReader().read(tmp_path / 'scan_para.hdf5')

        expected = parameters.product
        assert len(product.probe_positions) == len(expected.probe_positions)
        assert numpy.allclose(
            product.probe_positions.get_coordinates_x_m(),
            expected.probe_positions.get_coordinates_x_m(),
        )
        assert numpy.allclose(
            product.probe_positions.get_coordinates_y_m(),
            expected.probe_positions.get_coordinates_y_m(),
        )
        assert product.probes.get_array().shape == (1, 1, IMG_SIZE, IMG_SIZE)
        assert product.object_.get_pixel_geometry() == expected.object_.get_pixel_geometry()

    def test_either_member_reads_the_same_product(self, tmp_path: Path) -> None:
        save_training_data(tmp_path / 'scan.hdf5', _make_input())
        reader = PtychoFMProductFileReader()

        from_patterns = reader.read(tmp_path / 'scan_dp.hdf5')
        from_parameters = reader.read(tmp_path / 'scan_para.hdf5')

        assert numpy.array_equal(
            from_patterns.object_.get_array(), from_parameters.object_.get_array()
        )

    def test_positions_in_meters_are_taken_as_written(self, tmp_path: Path) -> None:
        # A 32-pixel object at 10 nm spans 320 nm, so a 3 nm scan is a hundredth of
        # the object and cannot be a pixel count.
        path = _write_para(tmp_path, x_m=numpy.array([0.0, 1e-9, 3e-9]))

        product = PtychoFMProductFileReader().read(path)

        assert product.probe_positions[2].x_m == pytest.approx(3e-9)

    def test_positions_in_pixels_are_scaled(self, tmp_path: Path) -> None:
        # A span of 16 against a 32-pixel object falls inside the trainer's window,
        # so the same numbers mean pixels and convert to meters.
        path = _write_para(
            tmp_path,
            x_m=numpy.array([0.0, 8.0, 16.0]),
            y_m=numpy.array([0.0, 5.0, 10.0]),
        )

        product = PtychoFMProductFileReader().read(path)

        assert product.probe_positions[2].x_m == pytest.approx(16.0 * PIXEL_SIZE_M)

    def test_one_axis_inside_the_window_is_not_enough(self, tmp_path: Path) -> None:
        # The trainer requires both axes to look like pixels before it treats either
        # as pixels, so a mixed pair has to read as meters.
        path = _write_para(
            tmp_path, x_m=numpy.array([0.0, 8.0, 16.0]), y_m=numpy.array([0.0, 1e-9, 2e-9])
        )

        product = PtychoFMProductFileReader().read(path)

        assert product.probe_positions[2].x_m == pytest.approx(16.0)

    def test_the_indexes_are_read_when_present(self, tmp_path: Path) -> None:
        path = _write_para(tmp_path, indexes=numpy.array([4, 9, 16]))

        product = PtychoFMProductFileReader().read(path)

        assert list(product.probe_positions.get_indexes()) == [4, 9, 16]

    def test_absent_indexes_default_to_a_running_count(self, tmp_path: Path) -> None:
        path = _write_para(tmp_path)

        product = PtychoFMProductFileReader().read(path)

        assert list(product.probe_positions.get_indexes()) == [0, 1, 2]

    def test_a_fold_slice_parameter_file_is_refused_clearly(self, tmp_path: Path) -> None:
        # fold_slice writes a '_para.hdf5' too, sharing the suffix and nothing else.
        path = tmp_path / 'data_roi0_para.hdf5'

        with h5py.File(path, 'w') as h5_file:
            h5_file.create_dataset('ppX', data=numpy.zeros(3))
            h5_file.create_dataset('ppY', data=numpy.zeros(3))

        with pytest.raises(ValueError, match='not a PtychoFM parameter file'):
            PtychoFMProductFileReader().read(path)

    def test_the_product_can_be_written_back_out(self, tmp_path: Path) -> None:
        save_training_data(tmp_path / 'scan.hdf5', _make_input())
        product = PtychoFMProductFileReader().read(tmp_path / 'scan_para.hdf5')

        save_product(tmp_path / 'product.h5', product)
        reloaded = load_product(tmp_path / 'product.h5')

        assert reloaded.object_.get_center() == product.object_.get_center()

    def test_both_readers_are_registered(self) -> None:
        # One artifact that is both the patterns and the reconstruction, so it is
        # offered as a scan to reconstruct and as a product to inspect.
        registry = PluginRegistry.load_plugins()

        assert registry.diffraction_file_readers.get_strategy_by_name(
            PtychoFMProductFileReader.SIMPLE_NAME
        )
        assert registry.product_file_readers.get_strategy_by_name(
            PtychoFMProductFileReader.SIMPLE_NAME
        )

    def test_the_writer_is_registered_under_the_library_name(self) -> None:
        # The shared name is what makes this the format offered by default whenever
        # PtychoFM is the selected backend.
        registry = PluginRegistry.load_plugins()

        assert registry.reconstructor_libraries.find_plugin(PtychoFMProductFileReader.SIMPLE_NAME)
        writer = registry.training_data_file_writers.get_strategy_by_name(
            PtychoFMProductFileReader.SIMPLE_NAME
        )
        assert isinstance(writer, PtychoFMTrainingDataFileWriter)


def _write_para(
    tmp_path: Path,
    *,
    x_m: numpy.ndarray | None = None,
    y_m: numpy.ndarray | None = None,
    indexes: numpy.ndarray | None = None,
) -> Path:
    """A minimal parameter file, for the cases the exporter cannot itself produce."""
    path = tmp_path / 'scan_para.hdf5'

    with h5py.File(path, 'w') as h5_file:
        obj = h5_file.create_dataset(
            ProductFileKeys.OBJECT_ARRAY, data=numpy.ones((1, 32, 32), dtype=complex)
        )
        obj.attrs[ProductFileKeys.OBJECT_PIXEL_HEIGHT] = PIXEL_SIZE_M
        obj.attrs[ProductFileKeys.OBJECT_PIXEL_WIDTH] = PIXEL_SIZE_M
        h5_file.create_dataset(
            ProductFileKeys.PROBE_ARRAY,
            data=numpy.ones((1, 1, IMG_SIZE, IMG_SIZE), dtype=complex),
        )
        h5_file.create_dataset(
            ProductFileKeys.PROBE_POSITION_X,
            data=numpy.zeros(NUM_POSITIONS) if x_m is None else x_m,
        )
        h5_file.create_dataset(
            ProductFileKeys.PROBE_POSITION_Y,
            data=numpy.zeros(NUM_POSITIONS) if y_m is None else y_m,
        )

        if indexes is not None:
            h5_file.create_dataset(ProductFileKeys.PROBE_POSITION_INDEXES, data=indexes)

    return path
