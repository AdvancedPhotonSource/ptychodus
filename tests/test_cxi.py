"""Tests for CXI file reading and writing in ptychodus.api.io and its plugin.

Foreign layouts are built by hand with h5py and named for the layout they follow,
so each test states exactly which fields a file holds.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import re

import h5py
import numpy
import pytest

from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_J, energy_eV_to_wavelength_m
from ptychodus.api.diffraction import (
    BeamCenter,
    DiffractionDatasetLayoutNode,
    DiffractionMetadata,
    Polarization,
    SimpleDiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import (
    CXIFileKeys,
    CXIPixelMask,
    CXIVariant,
    load_cxi_background,
    load_cxi_bad_pixels,
    load_cxi_diffraction_contents,
    load_cxi_flat_field,
    load_cxi_product,
    save_cxi_diffraction,
    save_cxi_product,
    save_cxi_reconstruct_input,
    save_product,
)
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput, prepare_reconstruct_input
from ptychodus.api.plugins import ProductProbePositionFileReader
from ptychodus.plugins.cxi_file import CXIDiffractionFileReader, CXIProductFileIO

from test_io import DISTINCTIVE_METADATA, _make_product, _make_reconstruct_input

ENERGY_EV = 8000.0
DISTANCE_M = 2.0
DETECTOR_PITCH_M = 75e-6


def _diffraction_dataset(
    *,
    background: numpy.ndarray | None = None,
    flat_field: numpy.ndarray | None = None,
) -> SimpleDiffractionDataset:
    rng = numpy.random.default_rng(3)
    arrays = [
        SimpleDiffractionArray(
            'first', numpy.array([5, 7, 9]), rng.integers(0, 100, (3, 6, 8)).astype(numpy.uint16)
        ),
        SimpleDiffractionArray(
            'second', numpy.array([20, 21]), rng.integers(0, 100, (2, 6, 8)).astype(numpy.uint16)
        ),
    ]
    metadata = DiffractionMetadata(
        num_patterns_per_array=[3, 2],
        pattern_dtype=numpy.dtype(numpy.uint16),
        detector_extent=ImageExtent(width_px=8, height_px=6),
        detector_distance_m=DISTANCE_M,
        detector_pixel_geometry=PixelGeometry(width_m=55e-6, height_m=75e-6),
        beam_center=BeamCenter(x_px=3, y_px=2),
        photon_energy_eV=ENERGY_EV,
        probe_photon_count=123456,
        exposure_time_s=0.25,
        tomography_angle_deg=12.5,
        tilt_angle_deg=-3.0,
        polarization=Polarization.LEFT_CIRCULAR,
    )
    bad_pixels = numpy.zeros((6, 8), dtype=bool)
    bad_pixels[1, 2] = True
    return SimpleDiffractionDataset(
        metadata,
        DiffractionDatasetLayoutNode.create_root(),
        arrays,
        bad_pixels,
        background=background,
        flat_field=flat_field,
    )


def _all_patterns(dataset: SimpleDiffractionDataset) -> numpy.ndarray:
    return numpy.concatenate([array.get_patterns() for array in dataset])


# ---------- diffraction round trips ----------


def test_diffraction_round_trips_every_metadata_field(tmp_path: Path) -> None:
    dataset = _diffraction_dataset()
    path = tmp_path / 'data.cxi'

    save_cxi_diffraction(path, dataset)
    loaded = CXIDiffractionFileReader().read(path)

    metadata = loaded.get_metadata()
    original = dataset.get_metadata()
    assert metadata.num_patterns_per_array == [5]
    assert metadata.pattern_dtype == numpy.uint16
    assert metadata.detector_extent == original.detector_extent
    assert metadata.detector_distance_m == pytest.approx(DISTANCE_M)
    assert metadata.detector_pixel_geometry == original.detector_pixel_geometry
    assert metadata.beam_center == original.beam_center
    assert metadata.photon_energy_eV == pytest.approx(ENERGY_EV)
    assert metadata.probe_photon_count == original.probe_photon_count
    assert metadata.exposure_time_s == pytest.approx(original.exposure_time_s)
    assert metadata.tomography_angle_deg == pytest.approx(original.tomography_angle_deg)
    assert metadata.tilt_angle_deg == pytest.approx(original.tilt_angle_deg)
    assert metadata.polarization == original.polarization
    numpy.testing.assert_array_equal(loaded[0].get_indexes(), [5, 7, 9, 20, 21])
    numpy.testing.assert_array_equal(loaded[0].get_patterns(), _all_patterns(dataset))
    numpy.testing.assert_array_equal(loaded.get_bad_pixels(), dataset.get_bad_pixels())


def test_detector_maps_round_trip(tmp_path: Path) -> None:
    background = numpy.full((6, 8), 1.5)
    flat_field = numpy.full((6, 8), 0.9)
    path = tmp_path / 'data.cxi'

    save_cxi_diffraction(path, _diffraction_dataset(background=background, flat_field=flat_field))
    loaded = CXIDiffractionFileReader().read(path)

    numpy.testing.assert_array_equal(loaded.get_background(), background)
    numpy.testing.assert_array_equal(loaded.get_flat_field(), flat_field)
    numpy.testing.assert_array_equal(load_cxi_background(path), background)
    numpy.testing.assert_array_equal(load_cxi_flat_field(path), flat_field)


# ---------- product round trips ----------


@pytest.mark.parametrize('variant', list(CXIVariant))
def test_every_variant_reads_back_the_whole_product(tmp_path: Path, variant: CXIVariant) -> None:
    product = _make_product(
        metadata=ProductMetadata(**DISTINCTIVE_METADATA),
        num_incoherent_modes=2,
        with_opr=True,
        with_layer_spacing=True,
        with_losses=True,
        with_position_photon_counts=True,
    )
    path = tmp_path / 'product.cxi'

    save_cxi_product(path, product, variant=variant)
    loaded = load_cxi_product(path)

    assert loaded.metadata == product.metadata
    numpy.testing.assert_array_equal(loaded.probes.get_array(), product.probes.get_array())
    numpy.testing.assert_array_equal(loaded.object_.get_array(), product.object_.get_array())
    assert loaded.object_.get_geometry() == product.object_.get_geometry()
    assert list(loaded.object_.layer_spacing_m) == list(product.object_.layer_spacing_m)
    assert loaded.probes.get_pixel_geometry() == product.probes.get_pixel_geometry()
    numpy.testing.assert_array_equal(
        loaded.probes.get_opr_weights(), product.probes.get_opr_weights()
    )
    numpy.testing.assert_array_equal(
        loaded.probe_positions.get_indexes(), product.probe_positions.get_indexes()
    )
    numpy.testing.assert_array_equal(
        loaded.probe_positions.get_coordinates_x_m(),
        product.probe_positions.get_coordinates_x_m(),
    )
    assert list(loaded.losses) == list(product.losses)


def test_sharp_layout_warns_when_it_drops_modes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    product = _make_product(num_incoherent_modes=2)
    path = tmp_path / 'product.cxi'

    with caplog.at_level('WARNING', logger='ptychodus.api.io'):
        save_cxi_product(path, product, variant=CXIVariant.SHARP)

    assert 'one probe mode' in caplog.text

    with h5py.File(path, 'r') as h5_file:
        entry = h5_file['entry_1']
        assert entry[CXIFileKeys.FINAL_ILLUMINATION].ndim == 2
        assert entry[CXIFileKeys.SOURCE_ILLUMINATION].ndim == 2
        assert entry['image_1/data'].ndim == 2
        assert CXIFileKeys.TITLE not in entry['image_1']


def test_cdtools_layout_adds_its_fields(tmp_path: Path) -> None:
    path = tmp_path / 'data.cxi'
    save_cxi_diffraction(
        path, _diffraction_dataset(flat_field=numpy.ones((6, 8))), variant=CXIVariant.CDTOOLS
    )

    with h5py.File(path, 'r') as h5_file:
        entry = h5_file['entry_1']
        assert entry[CXIFileKeys.DETECTOR_BASIS_VECTORS].shape == (3, 2)
        assert entry[CXIFileKeys.SOURCE_WAVELENGTH][()] == pytest.approx(
            energy_eV_to_wavelength_m(ENERGY_EV)
        )
        assert CXIFileKeys.DETECTOR_QE_MASK in entry

    # The transposed basis reads back to the same pitch.
    metadata = load_cxi_diffraction_contents(path).metadata
    assert metadata.detector_pixel_geometry == PixelGeometry(width_m=55e-6, height_m=75e-6)


# ---------- reconstruction input ----------


def test_reconstruct_input_zeroes_bad_pixels_by_default(tmp_path: Path) -> None:
    parameters = _make_reconstruct_input()
    path = tmp_path / 'input.cxi'

    save_cxi_reconstruct_input(path, parameters)

    with h5py.File(path, 'r') as h5_file:
        entry = h5_file['entry_1']
        patterns = entry['data_1/data'][()]
        mask = entry[CXIFileKeys.DETECTOR_MASK][()]

    bad = parameters.bad_pixels
    assert patterns.dtype == parameters.diffraction_patterns.dtype
    assert (patterns[:, bad] == 0).all()
    numpy.testing.assert_array_equal(patterns[:, ~bad], parameters.diffraction_patterns[:, ~bad])
    numpy.testing.assert_array_equal(mask[bad], int(CXIPixelMask.INVALID))
    assert not mask[~bad].any()


def test_inpainted_reconstruct_input_flags_repaired_pixels(tmp_path: Path) -> None:
    parameters = _make_reconstruct_input()
    path = tmp_path / 'input.cxi'

    save_cxi_reconstruct_input(path, parameters, inpaint_bad_pixels=True)

    with h5py.File(path, 'r') as h5_file:
        mask = h5_file['entry_1'][CXIFileKeys.DETECTOR_MASK][()]

    expected = int(CXIPixelMask.INVALID | CXIPixelMask.REPAIRED)
    numpy.testing.assert_array_equal(mask[parameters.bad_pixels], expected)


def test_reconstruct_input_derives_a_pitch_that_inverts_to_the_probe(tmp_path: Path) -> None:
    parameters = _make_reconstruct_input()
    path = tmp_path / 'input.cxi'

    save_cxi_reconstruct_input(path, parameters)
    detector = load_cxi_diffraction_contents(path).metadata.detector_pixel_geometry
    reloaded = load_cxi_product(path)

    assert detector is not None
    assert reloaded.probes.get_pixel_geometry().width_m == pytest.approx(
        parameters.product.probes.get_pixel_geometry().width_m
    )


def test_reconstruct_input_carries_its_detector_maps(tmp_path: Path) -> None:
    base = _make_reconstruct_input(detector_px=8)
    parameters = ReconstructInput(
        base.diffraction_patterns,
        base.bad_pixels,
        base.product,
        background=numpy.full((8, 8), 2.0),
        flat_field=numpy.full((8, 8), 0.5),
    )
    path = tmp_path / 'input.cxi'

    save_cxi_reconstruct_input(path, parameters)

    numpy.testing.assert_array_equal(load_cxi_background(path), 2.0)
    numpy.testing.assert_array_equal(load_cxi_flat_field(path), 0.5)


def test_exported_input_reassembles_and_pairs_every_pattern(tmp_path: Path) -> None:
    parameters = _make_reconstruct_input()
    path = tmp_path / 'input.cxi'
    save_cxi_reconstruct_input(path, parameters)

    data = assemble_dataset(CXIDiffractionFileReader().read(path), max_workers=1)
    product = load_cxi_product(path)
    reconstruct_input = prepare_reconstruct_input(data, product)

    assert len(reconstruct_input.diffraction_patterns) == len(parameters.diffraction_patterns)
    save_product(tmp_path / 'product.h5', product)


# ---------- standard conformance ----------


def _image_groups(entry: h5py.Group) -> list[h5py.Group]:
    return [entry[name] for name in entry if re.fullmatch(r'image_\d+', name)]


def test_standard_output_follows_cxi(tmp_path: Path) -> None:
    path = tmp_path / 'input.cxi'
    save_cxi_reconstruct_input(path, _make_reconstruct_input(num_incoherent_modes=2))

    with h5py.File(path, 'r') as h5_file:
        assert h5_file['cxi_version'][()] == 160
        assert h5_file['number_of_entries'][()] == 1
        entry = h5_file['entry_1']
        assert 'process_1' not in entry
        assert 'start_time' not in entry
        assert isinstance(entry['data_1/data'], h5py.Dataset)
        assert isinstance(entry.get('data_1/data', getlink=True), h5py.SoftLink)
        assert entry.get('data_2/data', getlink=True).path == '/entry_1/image_1/data'
        assert CXIFileKeys.SOURCE_ILLUMINATION not in entry
        assert CXIFileKeys.SOURCE_WAVELENGTH not in entry
        assert entry[CXIFileKeys.DETECTOR_MASK].dtype == numpy.uint32
        assert entry[CXIFileKeys.DETECTOR_BASIS_VECTORS].shape == (2, 3)
        assert entry[CXIFileKeys.DETECTOR_CORNER_POSITION].shape == (3,)

        titles = set()

        for image in _image_groups(entry):
            titles.add(image['title'][()].decode())
            assert image['data_type'][()].decode() in {
                'intensity',
                'electron density',
                'amplitude',
                'unphased amplitude',
                'autocorrelation',
            }
            assert image['data_space'][()].decode() in {'real', 'diffraction'}
            assert image['dimensionality'][()] == 2
            assert 'process_1/program' in image

            # Every name an axes attribute lists, other than y and x, exists alongside.
            axes = image['data'].attrs.get('axes')

            if axes is not None:
                for axis in str(axes).split(':'):
                    assert axis in {'y', 'x'} or axis in image

        assert titles == {'object', 'probe'}

        patterns_axes = entry[CXIFileKeys.DETECTOR_DATA].attrs['axes']
        assert patterns_axes == 'translation:y:x'
        assert 'translation' in entry['data_1']
        assert 'translation' in entry[CXIFileKeys.DETECTOR]


def test_product_only_files_point_data_1_at_the_object(tmp_path: Path) -> None:
    path = tmp_path / 'product.cxi'
    save_cxi_product(path, _make_product())

    with h5py.File(path, 'r') as h5_file:
        link = h5_file['entry_1'].get('data_1/data', getlink=True)
        assert link.path == '/entry_1/image_1/data'


def test_the_corner_position_follows_the_beam_center(tmp_path: Path) -> None:
    path = tmp_path / 'data.cxi'
    save_cxi_diffraction(path, _diffraction_dataset())

    with h5py.File(path, 'r') as h5_file:
        corner = h5_file['entry_1'][CXIFileKeys.DETECTOR_CORNER_POSITION][()]

    numpy.testing.assert_allclose(corner, [(3 + 0.5) * 55e-6, (2 + 0.5) * 75e-6, DISTANCE_M])


# ---------- foreign layouts ----------


def _write_file(path: Path, build: Callable[[h5py.Group], None]) -> Path:
    with h5py.File(path, 'w') as h5_file:
        h5_file['cxi_version'] = 140
        build(h5_file.create_group('entry_1'))

    return path


def _complex(shape: tuple[int, ...], seed: int = 0) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    return (rng.random(shape) + 1j * rng.random(shape)).astype(numpy.complex64)


def _write_reconstruction_layout_file(path: Path) -> Path:
    """Untitled object in image_1, final probe in its process group, no image_size."""

    def build(entry: h5py.Group) -> None:
        entry['image_1/data'] = _complex((40, 44))
        entry['image_1/process_1/final_illumination'] = _complex((16, 16), seed=1)
        entry['instrument_1/source_1/illumination'] = _complex((16, 16), seed=2)
        entry['instrument_1/source_1/energy'] = energy_eV_to_J(ENERGY_EV)
        entry['instrument_1/detector_1/data'] = numpy.ones((3, 16, 16), dtype=numpy.float32)
        entry['instrument_1/detector_1/distance'] = DISTANCE_M
        entry['instrument_1/detector_1/x_pixel_size'] = DETECTOR_PITCH_M
        entry['instrument_1/detector_1/y_pixel_size'] = DETECTOR_PITCH_M
        entry['instrument_1/detector_1/basis_vectors'] = [
            [0.0, -DETECTOR_PITCH_M, 0.0],
            [-DETECTOR_PITCH_M, 0.0, 0.0],
        ]
        entry['sample_1/geometry_1/translation'] = numpy.array(
            [[-1e-6, -2e-6, 0.0], [1e-6, -2e-6, 0.0], [1e-6, 2e-6, 0.0]], dtype=numpy.float32
        )

    return _write_file(path, build)


def test_a_reconstruction_layout_file_yields_object_probe_and_pitch(tmp_path: Path) -> None:
    path = _write_reconstruction_layout_file(tmp_path / 'recon.cxi')

    product = load_cxi_product(path)

    assert product.object_.get_array().shape == (1, 40, 44)
    with h5py.File(path, 'r') as h5_file:
        final = h5_file['entry_1/image_1/process_1/final_illumination'][()]
    numpy.testing.assert_array_equal(product.probes.get_array()[0, 0], final)
    # No image_size: the pitch comes from the far-field relation.
    expected = energy_eV_to_wavelength_m(ENERGY_EV) * DISTANCE_M / (16 * DETECTOR_PITCH_M)
    assert product.probes.get_pixel_geometry().width_m == pytest.approx(expected)
    assert product.object_.get_pixel_geometry().width_m == pytest.approx(expected)
    # No stored center: the object is centered on the scan.
    assert product.object_.get_center().x_m == pytest.approx(0.0, abs=1e-12)


def test_an_illumination_and_initial_image_file_is_read(tmp_path: Path) -> None:
    def build(entry: h5py.Group) -> None:
        entry['instrument_1/source_1/illumination'] = _complex((8, 8))
        entry['instrument_1/detector_1/initial_image'] = _complex((20, 20), seed=4)
        entry['sample_1/geometry_1/translation'] = numpy.zeros((2, 3))

    path = _write_file(tmp_path / 'input.cxi', build)
    product = load_cxi_product(path)

    assert product.probes.get_array().shape == (1, 1, 8, 8)
    assert product.object_.get_array().shape == (1, 20, 20)


def test_titled_images_are_found_in_either_order(tmp_path: Path) -> None:
    def build(entry: h5py.Group) -> None:
        entry['image_1/title'] = 'Probe'
        entry['image_1/data'] = _complex((8, 8))
        entry['image_2/title'] = 'object'
        entry['image_2/data'] = _complex((30, 30), seed=1)
        entry['sample_1/geometry_1/translation'] = numpy.zeros((1, 3))

    product = load_cxi_product(_write_file(tmp_path / 'spec.cxi', build))

    assert product.probes.get_array().shape[-2:] == (8, 8)
    assert product.object_.get_array().shape[-2:] == (30, 30)


def test_negated_probe_translations_land_where_the_patterns_were_taken(tmp_path: Path) -> None:
    """Translations stored as the negated probe position, with the default (3, 2) basis.

    A program that negates on read and uses that basis places probe k at pixel
    ``pinv(basis) @ (-T_k)``. ptychodus, reading the stored values unchanged, must
    move the probe across the object in the same direction.
    """
    pitch = 1e-7
    basis = numpy.array([[0.0, -pitch, 0.0], [-pitch, 0.0, 0.0]]).T  # columns: row, column
    translation = numpy.array([[0.0, 0.0, 0.0], [3 * pitch, 0.0, 0.0], [3 * pitch, 5 * pitch, 0.0]])

    def build(entry: h5py.Group) -> None:
        entry['instrument_1/detector_1/basis_vectors'] = basis
        entry['instrument_1/detector_1/mask'] = numpy.array([[0, 0x1000], [0x1, 0]], numpy.uint32)
        entry['instrument_1/source_1/wavelength'] = energy_eV_to_wavelength_m(ENERGY_EV)
        entry['instrument_1/detector_1/data'] = numpy.ones((3, 2, 2), dtype=numpy.float32)
        entry['data_1/data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')
        entry['sample_1/geometry_1/translation'] = translation
        entry['data_1/translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

    path = _write_file(tmp_path / 'transposed_basis.cxi', build)
    positions = load_cxi_product(path).probe_positions
    object_pixels = numpy.linalg.pinv(basis) @ (-translation).T  # rows: row, column

    product = _make_product(num_positions=len(positions))
    geometry = product.object_.get_geometry()
    columns = geometry.map_probe_positions_to_object_x_px(positions)
    rows = geometry.map_probe_positions_to_object_y_px(positions)
    pitch_x = geometry.pixel_width_m
    pitch_y = geometry.pixel_height_m
    numpy.testing.assert_allclose(
        numpy.diff(columns) * pitch_x, numpy.diff(object_pixels[1]) * pitch
    )
    numpy.testing.assert_allclose(numpy.diff(rows) * pitch_y, numpy.diff(object_pixels[0]) * pitch)

    contents = load_cxi_diffraction_contents(path)
    assert contents.metadata.photon_energy_eV == pytest.approx(ENERGY_EV)
    assert contents.metadata.detector_pixel_geometry == PixelGeometry(pitch, pitch)
    assert contents.bad_pixels is not None
    numpy.testing.assert_array_equal(contents.bad_pixels, [[False, False], [True, False]])
    strict = load_cxi_diffraction_contents(path, ignored_mask_bits=0)
    assert strict.bad_pixels is not None and strict.bad_pixels[0, 1]


def test_a_ptychodus_group_without_a_version_is_read_as_foreign(tmp_path: Path) -> None:
    def build(entry: h5py.Group) -> None:
        entry['ptychodus/probe_pixel_width_m'] = 1e-8
        entry['image_1/data'] = _complex((6, 6))
        entry['sample_1/geometry_1/translation'] = numpy.zeros((1, 3))

    product = load_cxi_product(_write_file(tmp_path / 'old.cxi', build))

    # The lone untitled image is the object, as in any foreign file.
    assert product.object_.get_array().shape[-2:] == (6, 6)


# ---------- fallbacks ----------


def _write_minimal_patterns_file(path: Path, **fields: object) -> Path:
    def build(entry: h5py.Group) -> None:
        entry['instrument_1/detector_1/data'] = numpy.ones((2, 4, 4), dtype=numpy.uint16)

        for key, value in fields.items():
            entry[key] = value

    return _write_file(path, build)


def test_absent_quantities_read_as_unknown(tmp_path: Path) -> None:
    contents = load_cxi_diffraction_contents(_write_minimal_patterns_file(tmp_path / 'm.cxi'))

    metadata = contents.metadata
    assert metadata.photon_energy_eV is None
    assert metadata.detector_distance_m is None
    assert metadata.detector_pixel_geometry is None
    assert contents.bad_pixels is None
    numpy.testing.assert_array_equal(contents.indexes, [0, 1])


def test_one_pixel_size_serves_both_axes(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'instrument_1/detector_1/x_pixel_size': 1e-4}
    )

    assert load_cxi_diffraction_contents(path).metadata.detector_pixel_geometry == PixelGeometry(
        1e-4, 1e-4
    )


@pytest.mark.parametrize(
    'basis',
    [
        numpy.array([[0.0, -2e-4, 0.0], [-1e-4, 0.0, 0.0]]),
        numpy.array([[0.0, -2e-4, 0.0], [-1e-4, 0.0, 0.0]]).T,
        numpy.array([[[0.0, -2e-4, 0.0], [-1e-4, 0.0, 0.0]]] * 2),
    ],
    ids=['rows', 'transposed', 'per-frame'],
)
def test_the_pitch_comes_from_basis_vectors_alone(tmp_path: Path, basis: numpy.ndarray) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'instrument_1/detector_1/basis_vectors': basis}
    )

    assert load_cxi_diffraction_contents(path).metadata.detector_pixel_geometry == PixelGeometry(
        1e-4, 2e-4
    )


def test_a_reoriented_detector_is_read_as_stored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi',
        **{'instrument_1/detector_1/basis_vectors': [[0.0, 1e-4, 0.0], [-1e-4, 0.0, 0.0]]},
    )

    with caplog.at_level('WARNING', logger='ptychodus.api.io'):
        load_cxi_diffraction_contents(path)

    assert 'default orientation' in caplog.text


def test_the_distance_comes_from_the_corner_along_the_normal(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi',
        **{
            'instrument_1/detector_1/corner_position': [1e-3, 2e-3, 1.5],
            'instrument_1/detector_1/basis_vectors': [[0.0, -1e-4, 0.0], [-1e-4, 0.0, 0.0]],
        },
    )

    assert load_cxi_diffraction_contents(path).metadata.detector_distance_m == pytest.approx(1.5)


def test_per_frame_quantities_reduce_to_their_median(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi',
        **{
            'instrument_1/detector_1/distance': [1.0, 2.0, 2.0],
            'instrument_1/source_1/energy': [energy_eV_to_J(ENERGY_EV)] * 2,
        },
    )

    metadata = load_cxi_diffraction_contents(path).metadata
    assert metadata.detector_distance_m == pytest.approx(2.0)
    assert metadata.photon_energy_eV == pytest.approx(ENERGY_EV)


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('instrument_1/beam_1/incident_energy', energy_eV_to_J(ENERGY_EV)),
        ('instrument_1/beam_1/incident_wavelength', energy_eV_to_wavelength_m(ENERGY_EV)),
    ],
)
def test_the_energy_falls_back_to_the_beam(tmp_path: Path, key: str, value: float) -> None:
    path = _write_minimal_patterns_file(tmp_path / 'm.cxi', **{key: value})

    assert load_cxi_diffraction_contents(path).metadata.photon_energy_eV == pytest.approx(ENERGY_EV)


def test_units_attributes_are_honored(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(tmp_path / 'm.cxi')

    with h5py.File(path, 'r+') as h5_file:
        distance = h5_file['entry_1'].create_dataset('instrument_1/detector_1/distance', data=500)
        distance.attrs['units'] = 'mm'

    assert load_cxi_diffraction_contents(path).metadata.detector_distance_m == pytest.approx(0.5)


def test_a_narrow_signed_mask_and_a_per_frame_mask_are_read(tmp_path: Path) -> None:
    mask = numpy.zeros((2, 4, 4), dtype=numpy.int8)
    mask[0, 0, 0] = -1
    mask[1, 3, 3] = 1
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'instrument_1/detector_1/mask': mask}
    )

    bad_pixels = load_cxi_diffraction_contents(path).bad_pixels

    assert bad_pixels is not None
    assert bad_pixels[0, 0] and bad_pixels[3, 3]
    assert bad_pixels.sum() == 2


def test_a_mask_of_the_wrong_shape_is_ignored(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'instrument_1/detector_1/mask': numpy.zeros((2, 2), numpy.uint32)}
    )

    assert load_cxi_diffraction_contents(path).bad_pixels is None


def test_a_single_frame_is_a_one_pattern_dataset(tmp_path: Path) -> None:
    def build(entry: h5py.Group) -> None:
        entry['instrument_1/detector_1/data'] = numpy.ones((4, 4), dtype=numpy.uint16)

    path = _write_file(tmp_path / 'one.cxi', build)

    assert CXIDiffractionFileReader().read(path)[0].get_patterns().shape == (1, 4, 4)


def test_a_dark_stack_averages_into_the_background(tmp_path: Path) -> None:
    dark = numpy.stack([numpy.full((4, 4), 1.0), numpy.full((4, 4), 3.0)])
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'instrument_1/detector_1/data_dark': dark}
    )

    numpy.testing.assert_array_equal(load_cxi_diffraction_contents(path).background, 2.0)


def test_a_qe_mask_is_read_as_the_flat_field(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'instrument_1/detector_1/qe_mask': numpy.full((4, 4), 0.8)}
    )

    numpy.testing.assert_array_equal(load_cxi_diffraction_contents(path).flat_field, 0.8)


def test_positions_load_from_a_file_without_a_probe(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'sample_1/geometry_1/translation': [[1e-6, 2e-6, 0.0]]}
    )

    positions = ProductProbePositionFileReader(CXIProductFileIO()).read(path)
    product = load_cxi_product(path)

    assert positions.get_coordinates_x_m()[0] == pytest.approx(1e-6)
    assert product.probes.width_px == 0
    assert product.object_.width_px == 0


def test_translations_in_the_wrong_orientation_are_refused(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi', **{'sample_1/geometry_1/translation': numpy.zeros((3, 10))}
    )

    with pytest.raises(ValueError, match=r'\(N, 2\)'):
        load_cxi_product(path).probe_positions


def test_mismatched_stored_indexes_fall_back_to_counting(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(
        tmp_path / 'm.cxi',
        **{
            'sample_1/geometry_1/translation': numpy.zeros((2, 3)),
            'ptychodus/probe_position_indexes': [4, 5, 6],
        },
    )

    numpy.testing.assert_array_equal(load_cxi_product(path).probe_positions.get_indexes(), [0, 1])


def test_another_entry_can_be_selected(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(tmp_path / 'm.cxi')

    with h5py.File(path, 'r+') as h5_file:
        h5_file['entry_2/instrument_1/detector_1/data'] = numpy.ones((5, 4, 4), numpy.uint16)

    assert load_cxi_diffraction_contents(path).metadata.num_patterns_per_array == [2]
    second = load_cxi_diffraction_contents(path, entry='entry_2')
    assert second.metadata.num_patterns_per_array == [5]


def test_map_readers_refuse_files_without_the_map(tmp_path: Path) -> None:
    path = _write_minimal_patterns_file(tmp_path / 'm.cxi')

    with pytest.raises(ValueError, match='mask'):
        load_cxi_bad_pixels(path)

    with pytest.raises(ValueError, match='background'):
        load_cxi_background(path)

    with pytest.raises(ValueError, match='flat field'):
        load_cxi_flat_field(path)


def test_a_mask_only_file_reads_as_bad_pixels(tmp_path: Path) -> None:
    def build(entry: h5py.Group) -> None:
        entry['instrument_1/detector_1/mask'] = numpy.array([[0, 2], [0x1000, 0]], numpy.uint32)

    path = _write_file(tmp_path / 'mask.cxi', build)

    numpy.testing.assert_array_equal(load_cxi_bad_pixels(path), [[False, True], [False, False]])


def test_the_bad_pixels_reader_agrees_with_the_diffraction_reader(tmp_path: Path) -> None:
    path = tmp_path / 'data.cxi'
    save_cxi_diffraction(path, _diffraction_dataset())

    numpy.testing.assert_array_equal(
        load_cxi_bad_pixels(path), CXIDiffractionFileReader().read(path).get_bad_pixels()
    )


def test_an_empty_product_round_trips(tmp_path: Path) -> None:
    """A product with no probe or object still makes a valid file, with data_1 present."""
    from ptychodus.api.object import Object
    from ptychodus.api.probe import ProbeEnsemble

    full = _make_product()
    product = Product(
        metadata=full.metadata,
        probe_positions=full.probe_positions,
        probes=ProbeEnsemble(None, None, None),
        object_=Object(None, None, None),
        losses=[],
    )
    path = tmp_path / 'empty.cxi'

    save_cxi_product(path, product)

    with h5py.File(path, 'r') as h5_file:
        assert 'data_1/data' in h5_file['entry_1']

    assert len(load_cxi_product(path).probe_positions) == len(full.probe_positions)
