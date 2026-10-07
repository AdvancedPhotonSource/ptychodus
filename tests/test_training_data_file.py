"""Round-trip tests for the ptychodus HDF5 training data format.

The writer lives in :mod:`ptychodus.api.io` and the readers in
:mod:`ptychodus.plugins.training_data_file`, so this is the only place the two
halves of the format meet. `TrainingDataFileKeys` is the vocabulary they share,
but nothing makes the writer emit what the readers expect -- these tests do.
"""

from pathlib import Path
import shutil

import h5py
import numpy
import numpy.testing
import pytest

from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import (
    TrainingDataFileKeys,
    save_product,
    save_training_data,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import compute_far_field_pixel_geometry
from ptychodus.api.reconstruct import ReconstructInput, prepare_reconstruct_input
from ptychodus.plugins.training_data_file import (
    TrainingDataDiffractionFileReader,
    TrainingDataProductFileReader,
)

NUM_POSITIONS = 6
DETECTOR_PX = 16
NUM_MODES = 3
OBJECT_PX = 40
OBJECT_PIXEL_SIZE_M = 1.2e-8
DETECTOR_DISTANCE_M = 2.0
PHOTON_ENERGY_EV = 10000.0
# Deliberately off-origin: positions are stored relative to the object center, so
# a zero-center fixture would hide the one lossy part of the round trip.
OBJECT_CENTER_X_M = 4.0e-7
OBJECT_CENTER_Y_M = -6.0e-7


def _make_reconstruct_input(*, with_bad_pixels: bool = True) -> ReconstructInput:
    rng = numpy.random.default_rng(11)

    metadata = ProductMetadata(
        name='fixture',
        comments='round trip',
        detector_distance_m=DETECTOR_DISTANCE_M,
        photon_energy_eV=PHOTON_ENERGY_EV,
        probe_photon_count=1234.0,
        exposure_time_s=0.25,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
    )
    positions = ProbePositionSequence(
        [ProbePosition(i, 3.0e-7 + i * 2.0e-8, -5.0e-7 + i * 3.0e-8) for i in range(NUM_POSITIONS)]
    )
    pixel_geometry = PixelGeometry(width_m=OBJECT_PIXEL_SIZE_M, height_m=OBJECT_PIXEL_SIZE_M)

    # Two OPR coherent modes, so taking index zero is an observable choice.
    probe_shape = (2, NUM_MODES, DETECTOR_PX, DETECTOR_PX)
    probes = ProbeSequence(
        array=rng.standard_normal(probe_shape) + 1j * rng.standard_normal(probe_shape),
        opr_weights=rng.standard_normal((NUM_POSITIONS, 2)),
        pixel_geometry=pixel_geometry,
    )
    object_shape = (1, OBJECT_PX, OBJECT_PX)
    object_ = Object(
        array=rng.standard_normal(object_shape) + 1j * rng.standard_normal(object_shape),
        pixel_geometry=pixel_geometry,
        center=ObjectCenter(x_m=OBJECT_CENTER_X_M, y_m=OBJECT_CENTER_Y_M),
        layer_spacing_m=[],
    )
    product = Product(
        metadata=metadata,
        probe_positions=positions,
        probes=probes,
        object_=object_,
        losses=[],
    )

    patterns = rng.integers(0, 1000, size=(NUM_POSITIONS, DETECTOR_PX, DETECTOR_PX)).astype(
        numpy.uint16
    )

    # Always a real mask, all-False when none is wanted: ReconstructInput requires
    # one, and nothing in production ever hands it None.
    bad_pixels = numpy.zeros((DETECTOR_PX, DETECTOR_PX), dtype=bool)
    if with_bad_pixels:
        bad_pixels[4:7, 4:7] = True

    return ReconstructInput(patterns, bad_pixels, product)


@pytest.fixture
def written(tmp_path: Path) -> tuple[Path, ReconstructInput]:
    parameters = _make_reconstruct_input()
    path = tmp_path / 'training.h5'
    save_training_data(path, parameters)
    return path, parameters


class TestTrainingDataRoundTrip:
    def test_object_and_probe_survive(self, written: tuple[Path, ReconstructInput]) -> None:
        """Arrays and the one recorded pixel size come back unchanged."""
        path, parameters = written
        product = TrainingDataProductFileReader().read(path)

        numpy.testing.assert_array_equal(
            product.object_.get_layer(0), parameters.product.object_.get_layer(0)
        )
        numpy.testing.assert_array_equal(
            product.probes.get_probe_no_opr().get_array(),
            parameters.product.probes.get_array()[0],
        )
        assert product.object_.get_pixel_geometry().width_m == OBJECT_PIXEL_SIZE_M
        assert product.object_.get_pixel_geometry().height_m == OBJECT_PIXEL_SIZE_M
        # Far field puts the probe on the object's sampling, which is why only
        # one pixel size is recorded.
        assert product.probes.get_pixel_geometry() == product.object_.get_pixel_geometry()

    def test_metadata_scalars_survive(self, written: tuple[Path, ReconstructInput]) -> None:
        """Energy and detector distance are carried, not defaulted to zero.

        A zero here loads as an empty product rather than an error, because the
        object pixel size the model layer re-derives from them gates the rebuild.
        """
        path, _ = written
        product = TrainingDataProductFileReader().read(path)

        assert product.metadata.photon_energy_eV == PHOTON_ENERGY_EV
        assert product.metadata.detector_distance_m == DETECTOR_DISTANCE_M

    def test_positions_reload_offset_by_the_object_center(
        self, written: tuple[Path, ReconstructInput]
    ) -> None:
        """The object center becomes the origin -- the documented loss."""
        path, parameters = written
        product = TrainingDataProductFileReader().read(path)
        original = parameters.product.probe_positions

        numpy.testing.assert_allclose(
            product.probe_positions.get_coordinates_x_m(),
            original.get_coordinates_x_m() - OBJECT_CENTER_X_M,
        )
        numpy.testing.assert_allclose(
            product.probe_positions.get_coordinates_y_m(),
            original.get_coordinates_y_m() - OBJECT_CENTER_Y_M,
        )

    def test_position_spacing_is_exact(self, written: tuple[Path, ReconstructInput]) -> None:
        """Only the origin shifts; the scan geometry itself is unchanged."""
        path, parameters = written
        product = TrainingDataProductFileReader().read(path)
        original = parameters.product.probe_positions

        numpy.testing.assert_allclose(
            numpy.diff(product.probe_positions.get_coordinates_x_m()),
            numpy.diff(original.get_coordinates_x_m()),
        )
        numpy.testing.assert_allclose(
            numpy.diff(product.probe_positions.get_coordinates_y_m()),
            numpy.diff(original.get_coordinates_y_m()),
        )

    def test_detector_pitch_is_recovered_by_the_far_field_inverse(
        self, written: tuple[Path, ReconstructInput]
    ) -> None:
        """The pitch the file cannot store inverts back to the object pixel size.

        The far-field relation is its own inverse, which is the whole reason the
        detector pixel size need not be written.
        """
        path, _ = written
        dataset = TrainingDataDiffractionFileReader().read(path)
        pitch = dataset.get_metadata().detector_pixel_geometry
        assert pitch is not None

        recovered = compute_far_field_pixel_geometry(
            pitch,
            ImageExtent(width_px=DETECTOR_PX, height_px=DETECTOR_PX),
            wavelength_m=energy_eV_to_wavelength_m(PHOTON_ENERGY_EV),
            propagation_distance_m=DETECTOR_DISTANCE_M,
        )

        assert recovered.width_m == pytest.approx(OBJECT_PIXEL_SIZE_M)
        assert recovered.height_m == pytest.approx(OBJECT_PIXEL_SIZE_M)

    def test_bad_pixel_mask_survives(self, written: tuple[Path, ReconstructInput]) -> None:
        """The real mask is read, not the all-false one the dataset would invent."""
        path, parameters = written
        dataset = TrainingDataDiffractionFileReader().read(path)

        numpy.testing.assert_array_equal(dataset.get_bad_pixels(), parameters.bad_pixels)
        assert dataset.get_bad_pixels().any()

    def test_patterns_keep_their_dtype_and_repair(
        self, written: tuple[Path, ReconstructInput]
    ) -> None:
        """Patterns reload as written: source dtype, masked positions filled."""
        path, parameters = written
        dataset = TrainingDataDiffractionFileReader().read(path)
        patterns = dataset[0].get_patterns()
        bad = parameters.bad_pixels
        assert bad is not None

        assert patterns.dtype == parameters.diffraction_patterns.dtype
        assert numpy.all(patterns[:, bad] != 0)

    def test_both_readers_agree_on_scan_indexes(
        self, written: tuple[Path, ReconstructInput]
    ) -> None:
        """Patterns pair to positions by index, so the two readers must agree.

        They are separate entry points over one file, and nothing but this holds
        them to the same numbering.
        """
        path, _ = written
        dataset = TrainingDataDiffractionFileReader().read(path)
        product = TrainingDataProductFileReader().read(path)

        expected = numpy.arange(NUM_POSITIONS)
        numpy.testing.assert_array_equal(dataset[0].get_indexes(), expected)
        numpy.testing.assert_array_equal(product.probe_positions.get_indexes(), expected)

    def test_loaded_product_can_be_written_back_out(
        self, written: tuple[Path, ReconstructInput], tmp_path: Path
    ) -> None:
        """The object carries a real center, so the product stays writable.

        A reader that leaves the center unset produces a product that can be
        reconstructed from but not saved, because save_product raises on it.
        """
        path, _ = written
        product = TrainingDataProductFileReader().read(path)

        out_path = tmp_path / 'resaved.h5'
        save_product(out_path, product)

        assert out_path.is_file()

    def test_pairs_every_pattern_to_a_position(
        self, written: tuple[Path, ReconstructInput]
    ) -> None:
        """End to end: the two halves reassemble into a usable reconstruct input."""
        path, _ = written
        dataset = TrainingDataDiffractionFileReader().read(path)
        product = TrainingDataProductFileReader().read(path)

        assembled = assemble_dataset(dataset)
        parameters = prepare_reconstruct_input(assembled, product)

        assert len(parameters.diffraction_patterns) == NUM_POSITIONS
        assert len(parameters.product.probe_positions) == NUM_POSITIONS

    @pytest.mark.parametrize('key', [str(member) for member in TrainingDataFileKeys])
    def test_every_written_key_is_consumed_by_a_reader(
        self, written: tuple[Path, ReconstructInput], tmp_path: Path, key: str
    ) -> None:
        """Removing any dataset breaks a reader, so none of them is dead weight.

        The writer and the readers sit in different layers; without this a key
        could be emitted and never read, or renamed on one side only.
        """
        path, _ = written
        damaged = tmp_path / f'without_{key}.h5'
        shutil.copyfile(path, damaged)

        with h5py.File(damaged, 'r+') as h5_file:
            del h5_file[key]

        readers = (TrainingDataDiffractionFileReader(), TrainingDataProductFileReader())
        failed = []

        for reader in readers:
            try:
                reader.read(damaged)
            except (KeyError, ValueError):
                failed.append(type(reader).__name__)

        assert failed, f'no reader reads "{key}"'


def test_absent_mask_round_trips_as_all_false(tmp_path: Path) -> None:
    """A product with no bad pixels still produces a readable file."""
    parameters = _make_reconstruct_input(with_bad_pixels=False)
    path = tmp_path / 'training.h5'
    save_training_data(path, parameters)

    dataset = TrainingDataDiffractionFileReader().read(path)

    assert not dataset.get_bad_pixels().any()
    numpy.testing.assert_array_equal(dataset[0].get_patterns(), parameters.diffraction_patterns)
