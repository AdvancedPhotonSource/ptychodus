"""Unit tests for ptychodus.api.io – diffraction and product HDF5 round-trips."""

from __future__ import annotations

from pathlib import Path

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    StandardFileLayout,
    load_diffraction_data,
    load_product,
    save_diffraction_data,
    save_product,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product, ProductMetadata
from ptychodus.api.reconstructor import AssembledDiffractionData


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
    with_opr: bool = False,
    with_layer_spacing: bool = False,
    with_losses: bool = False,
) -> Product:
    rng = numpy.random.default_rng(1)

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
        [ProbePosition(i, i * 1e-6, i * 2e-6) for i in range(num_positions)]
    )

    # Probe: shape (coherent=1 or 2, incoherent=1, height, width)
    num_coherent = 2 if with_opr else 1
    probe_array = rng.standard_normal(
        (num_coherent, 1, probe_height, probe_width)
    ) + 1j * rng.standard_normal((num_coherent, 1, probe_height, probe_width))
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
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
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

    def test_product_in_filename(self) -> None:
        assert StandardFileLayout.PRODUCT_IN == 'product-in.h5'

    def test_product_out_filename(self) -> None:
        assert StandardFileLayout.PRODUCT_OUT == 'product-out.h5'

    def test_settings_filename(self) -> None:
        assert StandardFileLayout.SETTINGS == 'settings.ini'

    def test_fluorescence_filenames(self) -> None:
        assert StandardFileLayout.FLUORESCENCE_IN == 'fluorescence-in.h5'
        assert StandardFileLayout.FLUORESCENCE_OUT == 'fluorescence-out.h5'

    def test_all_values_are_strings(self) -> None:
        for member in StandardFileLayout:
            assert isinstance(member.value, str)


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

    def test_mmap_raises_not_implemented(self, tmp_path: Path) -> None:
        original = _make_diffraction_data()
        file = tmp_path / 'diff.h5'
        save_diffraction_data(file, original)

        with pytest.raises(NotImplementedError):
            load_diffraction_data(file, mmap_file=tmp_path / 'mmap.bin')

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
        assert center.coordinate_x_m == pytest.approx(0.0)
        assert center.coordinate_y_m == pytest.approx(0.0)

    def test_probe_positions_preserved(self, tmp_path: Path) -> None:
        original = _make_product(num_positions=3)
        file = tmp_path / 'product.h5'

        save_product(file, original)
        loaded = load_product(file)

        assert len(loaded.probe_positions) == 3
        for i, (orig, load) in enumerate(zip(original.probe_positions, loaded.probe_positions)):
            assert load.index == orig.index
            assert load.coordinate_x_m == pytest.approx(orig.coordinate_x_m)
            assert load.coordinate_y_m == pytest.approx(orig.coordinate_y_m)

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
