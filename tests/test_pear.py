"""Unit tests for the PEAR reconstruction reader.

PEAR writes one flat HDF5 per checkpoint holding what pty-chi had in memory: the
object as a layer stack, the probe with its varying and incoherent axes, the refined
scan in object pixels, and a per-epoch loss. Three datasets appear only when the run
produced them, and the beam and detector appear in none of them.

The scan is the part that cannot be read off the names. It is row-major -- y first --
and measured about an origin pty-chi puts half a pixel away from ptychodus's on an
odd-sized axis, so both orderings and both parities are pinned below.
"""

from pathlib import Path

import h5py
import numpy
import pytest

from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import load_product, save_product
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeGeometry
from ptychodus.plugins.pear_product_file import PEARProductFileReader

OBJECT_PIXEL_SIZE_M = 1.5e-8
NUM_DETECTOR_PX = 64
NUM_POSITIONS = 4
PHOTON_ENERGY_EV = 8000.0

# The constants the plugin registers its reader with. Restated rather than imported so
# that a change to the registration has to be made deliberately in both places.
EIGER_PIXEL_SIZE_M = 75e-6


def _write(
    file_path: Path,
    *,
    num_layers: int = 1,
    num_varying_modes: int = 1,
    num_incoherent_modes: int = 2,
    object_height: int = 32,
    object_width: int = 32,
    positions_px: numpy.ndarray | None = None,
    with_slice_spacings: bool = False,
    with_opr_weights: bool = False,
    with_pos_corr: bool = False,
    with_loss: bool = True,
) -> Path:
    if positions_px is None:
        positions_px = numpy.zeros((NUM_POSITIONS, 2), dtype=numpy.float32)

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.create_dataset(
            'object',
            data=numpy.ones((num_layers, object_height, object_width), dtype=numpy.complex64),
        )
        h5_file.create_dataset(
            'probe',
            data=numpy.ones(
                (num_varying_modes, num_incoherent_modes, NUM_DETECTOR_PX, NUM_DETECTOR_PX),
                dtype=numpy.complex64,
            ),
        )
        h5_file.create_dataset('obj_pixel_size_m', data=OBJECT_PIXEL_SIZE_M)
        h5_file.create_dataset('positions_px', data=positions_px)
        h5_file.create_dataset(
            'init_positions_px', data=numpy.asarray(positions_px, dtype=numpy.float64)
        )

        if with_loss:
            h5_file.create_dataset('loss', data=numpy.array([3.0, 2.0, 1.0]))

        if with_slice_spacings:
            # Already one per gap, where fold_slice keeps one per layer plus one.
            h5_file.create_dataset(
                'slice_spacings_m', data=numpy.full(num_layers - 1, 1e-6, dtype=numpy.float64)
            )

        if with_opr_weights:
            h5_file.create_dataset(
                'opr_mode_weights',
                data=numpy.ones((NUM_POSITIONS, num_varying_modes), dtype=numpy.float32),
            )

        if with_pos_corr:
            group = h5_file.create_group('pos_corr')
            group.create_dataset('rotation', data=numpy.zeros(2))
            group.create_dataset('affine_matrix', data=numpy.zeros((2, 3), dtype=numpy.float32))

    return file_path


@pytest.fixture
def reader() -> PEARProductFileReader:
    return PEARProductFileReader(
        detector_pixel_size_m=EIGER_PIXEL_SIZE_M, photon_energy_eV=PHOTON_ENERGY_EV
    )


class TestPEARProductFileReader:
    def test_the_arrays_keep_pty_chi_axis_order(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        # pty-chi already holds both the way ptychodus does, so a transpose here would
        # be the bug rather than the fix.
        path = _write(
            tmp_path / 'recon_Niter100.h5',
            num_layers=3,
            num_varying_modes=4,
            num_incoherent_modes=2,
            with_slice_spacings=True,
        )

        product = reader.read(path)

        assert product.object_.get_array().shape == (3, 32, 32)
        assert product.probes.get_array().shape == (4, 2, NUM_DETECTOR_PX, NUM_DETECTOR_PX)

    def test_the_scan_is_row_major(self, reader: PEARProductFileReader, tmp_path: Path) -> None:
        # Column 0 is y. Reading it as x mirrors every scan about the diagonal, which
        # is the single most likely thing to get wrong here.
        positions = numpy.array([[2.0, 7.0]], dtype=numpy.float32)
        path = _write(tmp_path / 'recon_Niter100.h5', positions_px=positions)

        product = reader.read(path)

        assert product.probe_positions[0].y_m == pytest.approx(2.0 * OBJECT_PIXEL_SIZE_M)
        assert product.probe_positions[0].x_m == pytest.approx(7.0 * OBJECT_PIXEL_SIZE_M)

    def test_an_even_axis_needs_no_origin_shift(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        # round(N / 2) and N / 2 agree, so pty-chi's origin is ptychodus's.
        positions = numpy.array([[0.0, 0.0]], dtype=numpy.float32)
        path = _write(
            tmp_path / 'recon_Niter100.h5',
            object_height=32,
            object_width=32,
            positions_px=positions,
        )

        product = reader.read(path)

        assert product.probe_positions[0].x_m == pytest.approx(0.0)
        assert product.probe_positions[0].y_m == pytest.approx(0.0)

    def test_an_odd_axis_shifts_by_half_a_pixel(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        # pty-chi puts the origin at round(N / 2) - 0.5 and ptychodus at (N - 1) / 2,
        # which part company by half a pixel once the dimension is odd.
        positions = numpy.array([[0.0, 0.0]], dtype=numpy.float32)
        path = _write(
            tmp_path / 'recon_Niter100.h5',
            object_height=32,
            object_width=103,
            positions_px=positions,
        )

        product = reader.read(path)

        assert product.probe_positions[0].x_m == pytest.approx(0.5 * OBJECT_PIXEL_SIZE_M)
        assert product.probe_positions[0].y_m == pytest.approx(0.0)

    def test_the_slice_spacings_are_already_gaps(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        path = _write(tmp_path / 'recon_Niter100.h5', num_layers=3, with_slice_spacings=True)

        product = reader.read(path)

        assert list(product.object_.layer_spacing_m) == pytest.approx([1e-6, 1e-6])

    def test_the_opr_weights_are_read(self, reader: PEARProductFileReader, tmp_path: Path) -> None:
        path = _write(tmp_path / 'recon_Niter100.h5', num_varying_modes=3, with_opr_weights=True)

        product = reader.read(path)

        assert product.probes.get_opr_weights().shape == (NUM_POSITIONS, 3)

    def test_the_loss_is_one_value_per_epoch(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        product = reader.read(_write(tmp_path / 'recon_Niter100.h5'))

        assert [(loss.epoch, loss.value) for loss in product.losses] == [
            (0, 3.0),
            (1, 2.0),
            (2, 1.0),
        ]

    def test_the_conditional_datasets_may_all_be_absent(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        # A single-slice run without OPR or position correction writes none of them.
        path = _write(tmp_path / 'recon_Niter100.h5', with_loss=False)

        product = reader.read(path)

        assert product.probes.get_opr_weights_or_none() is None
        assert list(product.object_.layer_spacing_m) == []
        assert product.losses == []

    def test_the_position_correction_history_is_ignored(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        # A product has nowhere to put an affine history, and its presence must not
        # stop the rest of the file reading.
        path = _write(tmp_path / 'recon_Niter100.h5', with_pos_corr=True)

        assert len(reader.read(path).probe_positions) == NUM_POSITIONS

    def test_the_distance_is_recovered_from_the_sample_pixel_size(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        product = reader.read(_write(tmp_path / 'recon_Niter100.h5'))

        probe_geometry = ProbeGeometry.from_far_field(
            PixelGeometry(width_m=EIGER_PIXEL_SIZE_M, height_m=EIGER_PIXEL_SIZE_M),
            ImageExtent(width_px=NUM_DETECTOR_PX, height_px=NUM_DETECTOR_PX),
            wavelength_m=energy_eV_to_wavelength_m(PHOTON_ENERGY_EV),
            distance_m=product.metadata.detector_distance_m,
        )

        assert probe_geometry.pixel_width_m == pytest.approx(OBJECT_PIXEL_SIZE_M)

    def test_without_the_injected_geometry_the_distance_stays_zero(self, tmp_path: Path) -> None:
        # The file states neither, so a reader told neither reports what it knows.
        product = PEARProductFileReader().read(_write(tmp_path / 'recon_Niter100.h5'))

        assert product.metadata.detector_distance_m == 0.0
        assert product.metadata.photon_energy_eV == 0.0

    def test_the_registered_reader_carries_the_beamline_constants(self, tmp_path: Path) -> None:
        registry = PluginRegistry.load_plugins()
        registered = registry.product_file_readers.get_strategy_by_name(
            PEARProductFileReader.SIMPLE_NAME
        )

        product = registered.read(_write(tmp_path / 'recon_Niter100.h5'))

        assert product.metadata.photon_energy_eV == pytest.approx(PHOTON_ENERGY_EV)
        assert product.metadata.detector_distance_m > 0.0

    def test_the_product_can_be_written_back_out(
        self, reader: PEARProductFileReader, tmp_path: Path
    ) -> None:
        # save_product raises on an object with no center, so a reader that leaves one
        # unset produces a product that cannot round-trip.
        product = reader.read(_write(tmp_path / 'recon_Niter100.h5'))

        save_product(tmp_path / 'product.h5', product)
        reloaded = load_product(tmp_path / 'product.h5')

        assert reloaded.object_.get_center() == product.object_.get_center()
        assert reloaded.metadata.detector_distance_m == pytest.approx(
            product.metadata.detector_distance_m
        )
