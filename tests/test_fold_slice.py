"""Unit tests for the fold_slice plugin family.

The dataset names and shapes here are copied from the files the fold_slice preprocessing
step actually writes. Two layouts occur in practice and both have to read: a measured
scan stores the coordinates as a column vector, while a simulated one carries them flat.
"""

from pathlib import Path
from typing import Any

import h5py
import numpy
import pytest
import scipy.io

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import load_product, save_product
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionParseError
from ptychodus.plugins.fold_slice.position_file import FoldSlicePositionFileReader
from ptychodus.plugins.fold_slice._matfile import open_matlab_file
from ptychodus.plugins.fold_slice.product_file import FoldSliceProductFileReader

from matlab_file_fixtures import write_matlab_v73

WAVELENGTH_M = 1.549802e-10
OBJECT_PIXEL_SIZE_M = 1.884786e-08
NUM_POSITIONS = 4
NUM_DETECTOR_PX = 256

# The pitch the plugin registers its readers with. Restated here rather than imported so
# that a change to the registration has to be made deliberately in both places.
EIGER_PIXEL_SIZE_M = 75e-6


def _write_mat(file_path: Path, workspace: dict[str, Any], mat_format: str) -> Path:
    """Write one workspace in whichever MATLAB container *mat_format* names."""
    if mat_format == 'v73':
        return write_matlab_v73(file_path, workspace)

    scipy.io.savemat(file_path, workspace)
    return file_path


@pytest.fixture(params=['v5', 'v73'])
def mat_format(request: pytest.FixtureRequest) -> str:
    """Both containers MATLAB writes under the .mat extension.

    fold_slice has saved -v7.3 since 2023, so a fix proven only on the older container
    is a fix for files nobody produces any more.
    """
    return request.param


@pytest.fixture
def reader() -> FoldSlicePositionFileReader:
    return FoldSlicePositionFileReader()


def _write(file_path: Path, *, column_vectors: bool = False) -> Path:
    x_m = numpy.linspace(0.0, 3.0e-6, NUM_POSITIONS)
    y_m = numpy.linspace(-1.0e-6, 1.0e-6, NUM_POSITIONS)

    if column_vectors:
        x_m = x_m[:, numpy.newaxis]
        y_m = y_m[:, numpy.newaxis]

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset('ppX', data=x_m)
        h5_file.create_dataset('ppY', data=y_m)

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
        assert product.metadata.photon_energy_eV == pytest.approx(8000.0, rel=1e-5)

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
        assert product.metadata.photon_energy_eV == pytest.approx(8000.0, rel=1e-5)

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


class TestMatFile:
    """The two containers MATLAB writes under one extension, read into one shape.

    fold_slice switched its solvers to ``-v7.3`` in 2023, which is HDF5 and which the
    older parser refuses outright, so both have to read. What keeps the reconstruction
    reader free of a second code path is that the two produce the same thing, and that
    is a property worth asserting rather than assuming.
    """

    def _workspace(self) -> dict[str, Any]:
        rng = numpy.random.default_rng(1)
        # Four distinct axis lengths, so a transposed axis cannot pass unnoticed.
        probe = rng.random((4, 5, 2, 3)) + 1j * rng.random((4, 5, 2, 3))
        obj = rng.random((6, 7, 2)) + 1j * rng.random((6, 7, 2))
        return {
            'p': {
                'lambda': WAVELENGTH_M,
                # Deliberately anisotropic: equal components would hide an index swap.
                'dx_spec': numpy.array([1.1e-8, 2.2e-8]),
                'multi_slice_param': {'z_distance': numpy.array([1e-5, 2e-5, numpy.inf])},
            },
            'probe': probe.astype(numpy.complex64),
            'object': obj.astype(numpy.complex128),
            'outputs': {
                'probe_positions': rng.random((3, 2)),
                'fourier_error_out': numpy.array([numpy.nan, 1.0, 0.5]),
                'probe_evolution': rng.random((3, 3)),
            },
        }

    def _flatten(self, contents: Any, prefix: str = '') -> dict[str, Any]:
        flattened: dict[str, Any] = {}

        for key in contents:
            value = contents[key]

            if hasattr(value, 'keys'):
                flattened.update(self._flatten(value, f'{prefix}{key}.'))
            else:
                flattened[f'{prefix}{key}'] = value

        return flattened

    def test_the_two_formats_agree(self, tmp_path: Path) -> None:
        workspace = self._workspace()
        v5_path = _write_mat(tmp_path / 'v5.mat', workspace, 'v5')
        v73_path = _write_mat(tmp_path / 'v73.mat', workspace, 'v73')

        with open_matlab_file(v5_path) as v5, open_matlab_file(v73_path) as v73:
            from_v5 = self._flatten(v5)
            from_v73 = self._flatten(v73)

        assert sorted(from_v5) == sorted(from_v73)

        for key, expected in from_v5.items():
            actual = from_v73[key]
            assert type(actual) is type(expected), key
            assert numpy.shape(actual) == numpy.shape(expected), key
            assert numpy.asarray(actual).dtype == numpy.asarray(expected).dtype, key
            assert numpy.array_equal(actual, expected, equal_nan=True), key

    def test_a_one_by_one_variable_reads_as_a_python_scalar(self, tmp_path: Path) -> None:
        # MATLAB has no rank below two, so every scalar is stored as a 1x1 array and
        # has to come back as a number rather than as an array of one.
        path = _write_mat(tmp_path / 'scalar.mat', {'x': 2.5}, 'v73')

        with open_matlab_file(path) as contents:
            value = contents['x']

        assert value == pytest.approx(2.5)
        assert isinstance(value, float)

    def test_a_complex_array_keeps_its_precision(self, tmp_path: Path) -> None:
        # Recombining the stored real/imag pair by arithmetic would promote a single
        # to a double and quietly double the memory a probe occupies.
        values = numpy.ones((2, 3), dtype=numpy.complex64)
        path = _write_mat(tmp_path / 'single.mat', {'probe': values}, 'v73')

        with open_matlab_file(path) as contents:
            probe = contents['probe']

        assert probe.dtype == numpy.complex64

    def test_a_cell_array_reads_as_a_list(self, tmp_path: Path) -> None:
        # p.object_ROI and p.init_object_file are cells, so a file carrying them must
        # open even though the reconstruction reader never looks at them.
        path = _write_mat(tmp_path / 'cell.mat', {'p': {'object_ROI': [1.0, 2.0]}}, 'v73')

        with open_matlab_file(path) as contents:
            assert contents['p']['object_ROI'] == [1.0, 2.0]

    def test_an_unmodelled_class_reads_as_none_rather_than_raising(self, tmp_path: Path) -> None:
        # outputs carries serialized MATLAB objects in every real file. A file is not
        # unreadable merely for holding one.
        path = tmp_path / 'opaque.mat'
        _write_mat(path, {'p': {'lambda': WAVELENGTH_M}}, 'v73')

        with h5py.File(path, 'r+') as h5_file:
            handle = h5_file['p'].create_group('fnc')
            handle.attrs['MATLAB_class'] = numpy.bytes_(b'function_handle')

        with open_matlab_file(path) as contents:
            assert contents['p']['fnc'] is None
            assert contents['p']['lambda'] == pytest.approx(WAVELENGTH_M)

    def test_the_parser_metadata_is_not_a_workspace_variable(self, tmp_path: Path) -> None:
        # The older parser reports __header__ and friends; the newer container has no
        # equivalent, so they are dropped to make iteration agree as well as lookup.
        path = _write_mat(tmp_path / 'v5.mat', {'x': 1.0}, 'v5')

        with open_matlab_file(path) as contents:
            assert list(contents) == ['x']


class TestFoldSliceReconstructionVariants:
    """Layouts the solvers write that the reader used to mishandle.

    Each is derived from what the MATLAB source emits rather than from a file on hand:
    the multislice engine records its slice spacing whatever the layer count, a fly
    scan records one position table per probe mode, a variable probe records a weight
    matrix beside the basis, and the error history is NaN wherever nothing was
    measured. Every case runs against both containers, since a fix proven only on the
    pre-2023 one is a fix for files nobody writes any more.
    """

    def _write(
        self,
        file_path: Path,
        mat_format: str,
        *,
        num_modes: int = 2,
        num_varying_modes: int = 1,
        num_layers: int = 1,
        num_positions: int = 4,
        num_fly_modes: int = 0,
        multi_slice: bool = False,
        probe_evolution: numpy.ndarray | None = None,
        fourier_error_out: numpy.ndarray | None = None,
        dx_spec: numpy.ndarray | None = None,
    ) -> Path:
        shape: tuple[int, ...] = (NUM_DETECTOR_PX, NUM_DETECTOR_PX, num_modes)

        if num_varying_modes > 1:
            # MATLAB drops the trailing singleton, so the axis exists only above one.
            shape = (*shape, num_varying_modes)

        positions_shape: tuple[int, ...] = (num_positions, 2)

        if num_fly_modes:
            positions_shape = (*positions_shape, num_fly_modes)

        positions = numpy.arange(numpy.prod(positions_shape), dtype=float)
        p_struct: dict[str, Any] = {
            'lambda': WAVELENGTH_M,
            'dx_spec': numpy.array([OBJECT_PIXEL_SIZE_M, OBJECT_PIXEL_SIZE_M])
            if dx_spec is None
            else dx_spec,
        }

        if multi_slice:
            # The engine always writes a trailing free-space distance to the detector.
            spacings = [1e-5] * max(num_layers - 1, 0)
            p_struct['multi_slice_param'] = {'z_distance': numpy.array([*spacings, numpy.inf])}

        object_shape = (32, 32) if num_layers == 1 else (32, 32, num_layers)
        outputs: dict[str, Any] = {
            'probe_positions': positions.reshape(positions_shape),
            'fourier_error_out': numpy.zeros(4) if fourier_error_out is None else fourier_error_out,
        }

        if probe_evolution is not None:
            outputs['probe_evolution'] = probe_evolution

        return _write_mat(
            file_path,
            {
                'p': p_struct,
                'outputs': outputs,
                'probe': numpy.ones(shape, dtype=numpy.complex64),
                'object': numpy.ones(object_shape, dtype=numpy.complex64),
            },
            mat_format,
        )

    def _read(self, file_path: Path) -> Any:
        return FoldSliceProductFileReader(detector_pixel_size_m=EIGER_PIXEL_SIZE_M).read(file_path)

    def test_a_single_layer_multislice_object_reads(self, tmp_path: Path, mat_format: str) -> None:
        # The multislice engine records multi_slice_param whatever the layer count, and
        # MATLAB drops the layer axis at one -- so the object is 2-D and the spacing is
        # present, which used to index an axis that is not there.
        path = self._write(tmp_path / 'ms.mat', mat_format, num_layers=1, multi_slice=True)

        product = self._read(path)

        assert product.object_.layer_spacing_m == []
        assert product.object_.get_array().shape == (1, 32, 32)

    def test_a_multislice_object_keeps_its_gaps(self, tmp_path: Path, mat_format: str) -> None:
        path = self._write(tmp_path / 'ms3.mat', mat_format, num_layers=3, multi_slice=True)

        product = self._read(path)

        assert list(product.object_.layer_spacing_m) == pytest.approx([1e-5, 1e-5])

    def test_fly_scan_positions_take_the_first_mode(self, tmp_path: Path, mat_format: str) -> None:
        # A fly scan records one table per probe mode. They sample one trajectory, so
        # the first is the trajectory rather than a coordinate pair of arrays.
        path = self._write(tmp_path / 'fly.mat', mat_format, num_positions=4, num_fly_modes=3)

        product = self._read(path)

        assert len(product.probe_positions) == 4
        # Position i, axis j, mode 0 was written as (i * 2 + j) * 3.
        assert product.probe_positions[1].x_m == pytest.approx(6.0 * OBJECT_PIXEL_SIZE_M)
        assert product.probe_positions[1].y_m == pytest.approx(9.0 * OBJECT_PIXEL_SIZE_M)

    def test_opr_weights_are_attached(self, tmp_path: Path, mat_format: str) -> None:
        # The fourth probe axis is a basis for a position-dependent probe, not four
        # more coherent modes. Without probe_evolution it is loaded as the wrong thing.
        weights = numpy.linspace(0.0, 1.0, 4 * 3).reshape(4, 3)
        path = self._write(
            tmp_path / 'opr.mat', mat_format, num_varying_modes=3, probe_evolution=weights
        )

        product = self._read(path)

        assert product.probes.get_opr_weights().shape == (4, 3)
        assert product.probes.get_array().shape[0] == 3

    def test_a_single_opr_column_is_reshaped(self, tmp_path: Path, mat_format: str) -> None:
        # MATLAB squeezes a one-column weight matrix to a vector, which ProbeEnsemble
        # rejects; widening it must not transpose it into a single row.
        path = self._write(
            tmp_path / 'opr1.mat', mat_format, probe_evolution=numpy.ones(4, dtype=float)
        )

        product = self._read(path)

        assert product.probes.get_opr_weights().shape == (4, 1)

    def test_mismatched_opr_weights_are_dropped(
        self, tmp_path: Path, mat_format: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Weights that do not index the stored basis describe some other probe. Saying
        # so and keeping the unweighted basis beats refusing the file.
        path = self._write(
            tmp_path / 'bad.mat',
            mat_format,
            num_varying_modes=3,
            probe_evolution=numpy.ones((4, 2), dtype=float),
        )

        with caplog.at_level('WARNING'):
            product = self._read(path)

        assert product.probes.get_opr_weights_or_none() is None
        assert 'probe_evolution' in caplog.text

    def test_nan_losses_are_dropped_and_the_epochs_survive(
        self, tmp_path: Path, mat_format: str
    ) -> None:
        # The engine averages with omitnan and leaves NaN wherever nothing was
        # measured. Each value carries its own iteration, so a gap stays a gap.
        errors = numpy.array([numpy.nan, 1.0, numpy.nan, numpy.nan, 0.5])
        path = self._write(tmp_path / 'nan.mat', mat_format, fourier_error_out=errors)

        product = self._read(path)

        assert [(loss.epoch, loss.value) for loss in product.losses] == [(1, 1.0), (4, 0.5)]

    def test_an_all_nan_loss_history_reads_as_empty(self, tmp_path: Path, mat_format: str) -> None:
        # Whole runs record nothing. An empty history is honest where a thousand NaNs
        # would be copied into every product file written afterwards.
        path = self._write(
            tmp_path / 'allnan.mat', mat_format, fourier_error_out=numpy.full(8, numpy.nan)
        )

        assert self._read(path).losses == []

    def test_anisotropic_pixels_are_reported(
        self, tmp_path: Path, mat_format: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        # dx_spec is indexed by MATLAB's [rows columns], so the first component is the
        # height. fold_slice's own reader takes it as the width and the two agree in
        # every isotropic file, so the swap is kept and reported where it shows.
        path = self._write(
            tmp_path / 'aniso.mat', mat_format, dx_spec=numpy.array([1.1e-8, 2.2e-8])
        )

        with caplog.at_level('WARNING'):
            self._read(path)

        assert 'dx_spec' in caplog.text

    def test_an_opr_product_can_be_written_back_out(self, tmp_path: Path, mat_format: str) -> None:
        # save_product takes a different branch once opr_weights are present.
        weights = numpy.linspace(0.0, 1.0, 4 * 3).reshape(4, 3)
        path = self._write(
            tmp_path / 'opr.mat', mat_format, num_varying_modes=3, probe_evolution=weights
        )
        product = self._read(path)

        save_product(tmp_path / 'product.h5', product)
        reloaded = load_product(tmp_path / 'product.h5')

        assert numpy.allclose(reloaded.probes.get_opr_weights(), weights)
