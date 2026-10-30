from __future__ import annotations

from pathlib import Path
from typing import Final, Sequence
import logging

import numpy

from ptychodus.api.constants import wavelength_m_to_energy_eV
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.probe_positions import ProbePositionSequence, ProbePosition
from ptychodus.api.propagate import compute_far_field_propagation_distance
from ptychodus.api.typing import ComplexArrayType, NumberArrayType, RealArrayType
from ptychodus.api.reconstruct import LossValue

from ._matfile import open_matlab_file

logger = logging.getLogger(__name__)


def _conform_probe(probe_array: ComplexArrayType) -> ComplexArrayType:
    """Put the stored probe into ptychodus axis order, always four-dimensional.

    MATLAB holds the probe as ``[height, width, incoherent, varying]`` and drops a
    trailing singleton, so a run without a variable probe stores three dimensions and
    a single-mode one stores two. Restoring the axis here rather than leaving it to
    `ProbeEnsemble` means ``shape[0]`` is the varying-mode count in every case, which
    is what the weights below have to agree with.
    """
    if probe_array.ndim == 4:
        return probe_array.transpose(3, 2, 0, 1)

    if probe_array.ndim == 3:
        return probe_array.transpose(2, 0, 1)[numpy.newaxis, ...]

    return probe_array[numpy.newaxis, numpy.newaxis, ...]


def _conform_opr_weights(
    probe_evolution: NumberArrayType, num_varying_modes: int
) -> RealArrayType | None:
    """Condition ``outputs.probe_evolution`` into OPR weights, or None if it cannot be.

    The format stores the varying probe as a basis plus a weight per scan position, and
    the reconstruction recombines them by contracting the weight's second index against
    the basis. The weights are therefore meaningless without the basis they index, and
    a count that disagrees with it describes some other probe -- so say so and keep the
    unweighted basis rather than refuse the file.
    """
    weights = probe_evolution

    if numpy.iscomplexobj(weights):
        # Stored complex by some engines; the imaginary part is discarded where
        # fold_slice itself plots these.
        weights = weights.real

    real_weights = numpy.reshape(weights, (weights.shape[0], -1)).astype(numpy.float64)

    if real_weights.shape[1] != num_varying_modes:
        logger.warning(
            'Discarding probe_evolution: %d weight columns describe %d probe modes.',
            real_weights.shape[1],
            num_varying_modes,
        )
        return None

    return real_weights


def _conform_positions_px(probe_positions: RealArrayType) -> RealArrayType:
    """Reduce the stored positions to one ``(N, 2)`` table.

    A fly scan records a separate table per probe mode. They describe one trajectory
    sampled by each mode, so the first is the trajectory.
    """
    if probe_positions.ndim > 2:
        return probe_positions[..., 0]

    return probe_positions


def _read_losses(fourier_error_out: RealArrayType) -> list[LossValue]:
    """Read the loss history, dropping the iterations that recorded no error.

    The engine averages across positions with ``omitnan`` and leaves the entry NaN on
    every iteration where nothing was measured, which is most of them -- whole runs are
    NaN throughout. Each value carries its own iteration number, so dropping one leaves
    a gap on a real axis rather than renumbering what survives.
    """
    losses: list[LossValue] = []

    for epoch, value in enumerate(numpy.atleast_1d(fourier_error_out)):
        if numpy.isfinite(value):
            losses.append(LossValue(epoch, float(value)))

    return losses


class FoldSliceProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'fold_slice_mat'

    def __init__(self, detector_pixel_size_m: float | None = None) -> None:
        """Read a fold_slice reconstruction.

        `detector_pixel_size_m` stands in for a pitch the file does not record, and is
        what lets the sample-to-detector distance be recovered from the sample pixel
        size the file does record. Without it the distance reads as zero, which leaves
        the product unusable anywhere the object sampling has to be re-derived.
        """
        self._detector_pixel_size_m = detector_pixel_size_m

    def get_file_description(self) -> str:
        return 'fold_slice Files'

    def get_file_extensions(self) -> set[str]:
        return {'.mat'}

    def _compute_detector_distance_m(
        self,
        object_pixel_size_m: float,
        probe_extent: ImageExtent,
        wavelength_m: float | None,
    ) -> float:
        """Recover the sample-to-detector distance, or zero when the inputs are missing.

        The file states the sample pixel size rather than the distance, and the two are
        the same fact under the far-field relation once the detector pitch and the
        pattern width are known. The probe supplies the width: it is stored on the
        detector grid, so its array is as wide as the patterns were.

        Zero when the wavelength or the detector pitch is unavailable, matching what
        this reported before the distance could be recovered at all.
        """
        if wavelength_m is None or self._detector_pixel_size_m is None:
            return 0.0

        detector_pixel_geometry = PixelGeometry(
            width_m=self._detector_pixel_size_m, height_m=self._detector_pixel_size_m
        )
        return compute_far_field_propagation_distance(
            detector_pixel_geometry,
            probe_extent,
            wavelength_m=wavelength_m,
            conjugate_pixel_width_m=object_pixel_size_m,
        )

    def read(self, file_path: Path) -> Product:
        point_list: list[ProbePosition] = list()

        with open_matlab_file(file_path) as mat_dict:
            p_struct = mat_dict['p']

            try:
                wavelength_m = p_struct['lambda']
            except KeyError:
                wavelength_m = None
                photon_energy_eV = 0.0  # noqa: N806
            else:
                photon_energy_eV = wavelength_m_to_energy_eV(wavelength_m)  # noqa: N806

            try:
                tomography_angle_deg = p_struct['angle']
            except KeyError:
                tomography_angle_deg = 0.0

            dx_spec = p_struct['dx_spec']
            pixel_width_m = dx_spec[0]
            pixel_height_m = dx_spec[1]

            if pixel_width_m != pixel_height_m:
                # `p.dx_spec` is indexed by MATLAB's `p.asize`, which is [rows columns],
                # so the first component is the height. Reading it as the width matches
                # what fold_slice's own round-trip reader does, and the two components
                # are equal in every file that is not a tilted-sample scan -- so the
                # swap is kept for agreement and reported where it can be observed.
                logger.warning(
                    'Anisotropic dx_spec %g x %g is read width-first, as fold_slice does.',
                    pixel_width_m,
                    pixel_height_m,
                )

            pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)

            probe_array = _conform_probe(mat_dict['probe'])

            metadata = ProductMetadata(
                name=file_path.stem,
                comments='',
                detector_distance_m=self._compute_detector_distance_m(
                    pixel_width_m,
                    ImageExtent(width_px=probe_array.shape[-1], height_px=probe_array.shape[-2]),
                    wavelength_m,
                ),
                photon_energy_eV=photon_energy_eV,
                probe_photon_count=0.0,  # not included in file
                exposure_time_s=0.0,  # not included in file
                mass_attenuation_m2_per_kg=0.0,  # not included in file
                tomography_angle_deg=tomography_angle_deg,
            )

            outputs_struct = mat_dict['outputs']
            probe_positions = _conform_positions_px(outputs_struct['probe_positions'])

            for idx, pos_px in enumerate(probe_positions):
                point = ProbePosition(
                    idx,
                    pos_px[0] * pixel_width_m,
                    pos_px[1] * pixel_height_m,
                )
                point_list.append(point)

            try:
                probe_evolution = outputs_struct['probe_evolution']
            except KeyError:
                opr_weights = None
            else:
                opr_weights = _conform_opr_weights(probe_evolution, probe_array.shape[0])

            probe = ProbeEnsemble(
                array=probe_array,
                opr_weights=opr_weights,
                pixel_geometry=pixel_geometry,
            )

            object_array = mat_dict['object']

            if object_array.ndim == 3:
                # object_array[height, width, num_layers]
                object_array = object_array.transpose(2, 0, 1)

            layer_spacing_m: Sequence[float] = list()

            try:
                multi_slice_param = p_struct['multi_slice_param']
                z_distance = multi_slice_param['z_distance']
            except KeyError:
                pass
            else:
                # The multislice engine records this whatever the layer count, and a
                # single-layer run has no layer axis at all. `z_distance` ends with the
                # free-space distance to the detector, so the gaps are what precedes it.
                num_layers = object_array.shape[-3] if object_array.ndim > 2 else 1
                spacings = numpy.atleast_1d(numpy.squeeze(z_distance))
                layer_spacing_m = [float(spacing) for spacing in spacings[: num_layers - 1]]

            object_ = Object(
                array=object_array,
                pixel_geometry=pixel_geometry,
                # The format stores probe_positions as pixel offsets about the object
                # array's own center, so that center is the origin of the frame the
                # positions above were converted into and this is not a placeholder.
                # Leaving it unset instead makes the product unwritable: save_product
                # raises on an object with no center.
                center=ObjectCenter(x_m=0.0, y_m=0.0),
                layer_spacing_m=layer_spacing_m,
            )

            try:
                fourier_error_out = outputs_struct['fourier_error_out']
            except KeyError:
                losses: list[LossValue] = []
            else:
                losses = _read_losses(fourier_error_out)

            return Product(
                metadata=metadata,
                probe_positions=ProbePositionSequence(point_list),
                probes=probe,
                object_=object_,
                losses=losses,
            )
