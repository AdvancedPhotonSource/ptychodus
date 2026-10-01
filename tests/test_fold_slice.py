"""Unit tests for the fold_slice plugin family.

The dataset names and shapes here are copied from the files the fold_slice preprocessing
step actually writes. Two layouts occur in practice and both have to read: a measured
scan records ``/lambda``, ``/dx`` and ``/angle`` beside the coordinates and stores the
coordinates as a column vector, while a simulated one carries flat ``/ppX`` and ``/ppY``
and nothing else. The geometry is what lets a product be built without the experiment
being described on the command line, so a file that omits it must read as omitted rather
than as zero.
"""

from pathlib import Path

import h5py
import numpy
import pytest
import scipy.io

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import load_product, save_product
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.fold_slice._pairing import find_position_file
from ptychodus.plugins.fold_slice.position_file import (
    FoldSlicePositionFileReader,
    read_fold_slice_parameters,
)
from ptychodus.plugins.fold_slice.product_file import FoldSliceProductFileReader

WAVELENGTH_M = 1.549802e-10
OBJECT_PIXEL_SIZE_M = 1.884786e-08
TOMOGRAPHY_ANGLE_DEG = 17.5
NUM_POSITIONS = 4
NUM_DETECTOR_PX = 256

# The pitch the plugin registers its readers with. Restated here rather than imported so
# that a change to the registration has to be made deliberately in both places.
EIGER_PIXEL_SIZE_M = 75e-6


@pytest.fixture
def reader() -> FoldSlicePositionFileReader:
    return FoldSlicePositionFileReader()


def _write(file_path: Path, *, column_vectors: bool = False, with_geometry: bool = False) -> Path:
    x_m = numpy.linspace(0.0, 3.0e-6, NUM_POSITIONS)
    y_m = numpy.linspace(-1.0e-6, 1.0e-6, NUM_POSITIONS)

    if column_vectors:
        x_m = x_m[:, numpy.newaxis]
        y_m = y_m[:, numpy.newaxis]

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('ppX', data=x_m)
        h5_file.create_dataset('ppY', data=y_m)

        if with_geometry:
            # One-element datasets, as the preprocessing step writes them.
            h5_file.create_dataset('lambda', data=numpy.array([WAVELENGTH_M]))
            h5_file.create_dataset('dx', data=numpy.array([OBJECT_PIXEL_SIZE_M]))
            h5_file.create_dataset('angle', data=numpy.array([TOMOGRAPHY_ANGLE_DEG]))

    return file_path


def test_flat_coordinates_read(reader: FoldSlicePositionFileReader, tmp_path: Path) -> None:
    positions = reader.read(_write(tmp_path / 'sim_para.hdf5'))

    assert len(positions) == NUM_POSITIONS
    assert positions[0].x_m == pytest.approx(0.0)
    assert positions[-1].x_m == pytest.approx(3.0e-6)


def test_column_vector_coordinates_read(
    reader: FoldSlicePositionFileReader, tmp_path: Path
) -> None:
    """A measured scan stores (N, 1) rather than (N,); squeezing is what makes both work."""
    positions = reader.read(_write(tmp_path / 'measured_para.hdf5', column_vectors=True))

    assert len(positions) == NUM_POSITIONS
    assert positions[-1].y_m == pytest.approx(1.0e-6)


def test_mismatched_coordinate_arrays_are_rejected(
    reader: FoldSlicePositionFileReader, tmp_path: Path
) -> None:
    file_path = tmp_path / 'ragged_para.hdf5'

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('ppX', data=numpy.zeros(NUM_POSITIONS))
        h5_file.create_dataset('ppY', data=numpy.zeros(NUM_POSITIONS + 1))

    with pytest.raises(ProbePositionParseError):
        reader.read(file_path)


def test_a_measured_parameter_file_yields_its_geometry(tmp_path: Path) -> None:
    parameters = read_fold_slice_parameters(
        _write(tmp_path / 'measured_para.hdf5', column_vectors=True, with_geometry=True)
    )

    assert parameters.probe_wavelength_m == pytest.approx(WAVELENGTH_M)
    assert parameters.object_pixel_size_m == pytest.approx(OBJECT_PIXEL_SIZE_M)
    assert parameters.tomography_angle_deg == pytest.approx(TOMOGRAPHY_ANGLE_DEG)


def test_a_simulated_parameter_file_yields_nothing(tmp_path: Path) -> None:
    # None rather than 0.0: a zero probe energy or detector distance is a real value that
    # would collapse the sample-plane pixel size, so "absent" must stay distinguishable.
    parameters = read_fold_slice_parameters(_write(tmp_path / 'sim_para.hdf5'))

    assert parameters.probe_wavelength_m is None
    assert parameters.object_pixel_size_m is None
    assert parameters.tomography_angle_deg is None


def test_an_unexpected_shape_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    file_path = _write(tmp_path / 'odd_para.hdf5')

    with h5py.File(file_path, 'a') as h5_file:
        h5_file.create_dataset('lambda', data=numpy.array([1.0e-10, 2.0e-10]))
        h5_file.create_dataset('dx', data=numpy.array([OBJECT_PIXEL_SIZE_M]))

    parameters = read_fold_slice_parameters(file_path)

    assert parameters.probe_wavelength_m is None
    assert parameters.object_pixel_size_m == pytest.approx(OBJECT_PIXEL_SIZE_M)


@pytest.mark.parametrize(
    'diffraction_name, position_name',
    [
        ('data_roi0_dp.hdf5', 'data_roi0_para.hdf5'),
        ('data_roi0_Ndp256_rs1_dp.h5', 'data_roi0_Ndp256_rs1_para.h5'),
        # "_dp" only has to end the stem, so a scan whose own name contains it is fine.
        ('fly145_dp_dp.hdf5', 'fly145_dp_para.hdf5'),
    ],
)
def test_the_para_companion_is_derived_from_the_dp_name(
    diffraction_name: str, position_name: str
) -> None:
    assert find_position_file(Path('/scans') / diffraction_name) == Path('/scans') / position_name


@pytest.mark.parametrize('name', ['data_roi0.hdf5', 'data_roi0_para.hdf5', 'dp_data.hdf5'])
def test_a_name_off_the_convention_derives_no_companion(name: str) -> None:
    assert find_position_file(Path('/scans') / name) is None


class TestFoldSliceProductFileReader:
    """The .mat reconstruction output, and the distance it implies but does not state.

    The format records the sample pixel size (``p.dx_spec``) and the wavelength
    (``p.lambda``) but no detector distance -- ``p.detector`` carries only a binning
    flag and ``outputs.z_distance`` is the near-field propagation distance, which is
    ``inf`` for a far-field scan. The distance is therefore recovered from the pixel
    size, which needs a detector pitch the reader is told at registration.
    """

    def _write(self, file_path: Path, *, num_modes: int = 2) -> Path:
        probe = numpy.ones((NUM_DETECTOR_PX, NUM_DETECTOR_PX, num_modes), dtype=numpy.complex128)
        scipy.io.savemat(
            file_path,
            {
                'p': {
                    'lambda': WAVELENGTH_M,
                    'dx_spec': numpy.array([OBJECT_PIXEL_SIZE_M, OBJECT_PIXEL_SIZE_M]),
                },
                'outputs': {
                    'probe_positions': numpy.zeros((3, 2)),
                    'fourier_error_out': numpy.zeros(4, dtype=numpy.float32),
                },
                'probe': probe,
                'object': numpy.ones((32, 32), dtype=numpy.complex64),
            },
        )
        return file_path

    def test_the_distance_is_recovered_from_the_sample_pixel_size(self, tmp_path: Path) -> None:
        reader = FoldSliceProductFileReader(detector_pixel_size_m=EIGER_PIXEL_SIZE_M)

        product = reader.read(self._write(tmp_path / 'Niter100.mat'))

        assert product.metadata.detector_distance_m == pytest.approx(2.335, rel=1e-5)
        assert product.metadata.probe_energy_eV == pytest.approx(8000.0, rel=1e-5)

    def test_the_recovered_distance_reproduces_the_object_sampling(self, tmp_path: Path) -> None:
        reader = FoldSliceProductFileReader(detector_pixel_size_m=EIGER_PIXEL_SIZE_M)
        product = reader.read(self._write(tmp_path / 'Niter100.mat'))

        probe_geometry = ProbeGeometry.from_far_field(
            PixelGeometry(width_m=EIGER_PIXEL_SIZE_M, height_m=EIGER_PIXEL_SIZE_M),
            ImageExtent(width_px=NUM_DETECTOR_PX, height_px=NUM_DETECTOR_PX),
            wavelength_m=WAVELENGTH_M,
            distance_m=product.metadata.detector_distance_m,
        )

        assert probe_geometry.pixel_width_m == pytest.approx(OBJECT_PIXEL_SIZE_M)

    def test_without_a_detector_pitch_the_distance_stays_zero(self, tmp_path: Path) -> None:
        # The pitch is the one input the file cannot supply, so a registration that
        # names no instrument reads exactly as it did before the recovery existed.
        reader = FoldSliceProductFileReader()

        product = reader.read(self._write(tmp_path / 'Niter100.mat'))

        assert product.metadata.detector_distance_m == 0.0
        assert product.metadata.probe_energy_eV == pytest.approx(8000.0, rel=1e-5)

    def test_the_registered_reader_carries_the_eiger_pitch(self, tmp_path: Path) -> None:
        # What the family is actually pointed at: every instrument writing this layout
        # reads an Eiger, so the registered reader recovers a real distance rather than
        # the zero a reader told no pitch reports.
        registry = PluginRegistry.load_plugins()
        reader = registry.product_file_readers.get_strategy_by_name(
            FoldSliceProductFileReader.SIMPLE_NAME
        )

        product = reader.read(self._write(tmp_path / 'Niter100.mat'))

        assert product.metadata.detector_distance_m == pytest.approx(2.335, rel=1e-5)

    def test_the_product_can_be_written_back_out(self, tmp_path: Path) -> None:
        # save_product raises on an object with no center, so a reader that leaves one
        # unset produces a product that cannot round-trip. The format centers its
        # probe_positions on the object array, which makes the origin the real answer.
        reader = FoldSliceProductFileReader(detector_pixel_size_m=EIGER_PIXEL_SIZE_M)
        product = reader.read(self._write(tmp_path / 'Niter100.mat'))

        save_product(tmp_path / 'product.h5', product)
        reloaded = load_product(tmp_path / 'product.h5')

        assert reloaded.object_.get_center() == product.object_.get_center()
        assert reloaded.metadata.detector_distance_m == pytest.approx(
            product.metadata.detector_distance_m
        )
