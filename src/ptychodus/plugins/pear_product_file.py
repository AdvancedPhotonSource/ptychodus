"""Read the HDF5 reconstruction PEAR writes.

PEAR saves one ``recon_Niter<N>.h5`` per checkpoint, flat at the root of the run
directory: the object and probe as pty-chi holds them, the refined scan in object
pixels, the per-epoch loss, and whichever of the slice spacings, OPR weights and
position-correction history the run produced. It records nothing about the beam or
the detector -- those live only in the sibling parameter file PEAR was given -- so
the two values needed to place the object in space are supplied at registration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import h5py
import numpy

from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import LossValue, Product, ProductFileReader, ProductMetadata
from ptychodus.api.propagate import compute_far_field_propagation_distance

# Dataset names, flat at the root. `pos_corr` is a group holding the affine history
# of the position correction, which has no place in a product and is not read.
_OBJECT_ARRAY = 'object'
_PROBE_ARRAY = 'probe'
_LOSS = 'loss'
_POSITIONS_PX = 'positions_px'
_OBJECT_PIXEL_SIZE = 'obj_pixel_size_m'
_SLICE_SPACINGS = 'slice_spacings_m'
_OPR_MODE_WEIGHTS = 'opr_mode_weights'


def _read_positions(
    positions_px: numpy.ndarray, object_extent_px: tuple[int, int], pixel_size_m: float
) -> ProbePositionSequence:
    """Convert the stored scan into the object frame ptychodus measures positions in.

    The table is row-major -- y in the first column -- and in object pixels about an
    origin pty-chi puts at ``round(N / 2) - 0.5``. ptychodus puts the center of the
    array at ``(N - 1) / 2``, so the two differ by ``round(N / 2) - N / 2``: nothing
    for an even dimension and half a pixel for an odd one.
    """
    height_px, width_px = object_extent_px
    offset_y_px = numpy.round(height_px / 2.0) - height_px / 2.0
    offset_x_px = numpy.round(width_px / 2.0) - width_px / 2.0

    return ProbePositionSequence(
        [
            ProbePosition(
                index,
                float((position[1] + offset_x_px) * pixel_size_m),
                float((position[0] + offset_y_px) * pixel_size_m),
            )
            for index, position in enumerate(positions_px)
        ]
    )


class PEARProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'PEAR'

    def __init__(
        self,
        detector_pixel_size_m: float | None = None,
        photon_energy_eV: float = 0.0,  # noqa: N803
    ) -> None:
        """Read a PEAR reconstruction.

        The file records the sample pixel size but neither the beam nor the detector,
        and those are the same fact under the far-field relation. So both stand in for
        what the file omits, and together they recover the sample-to-detector distance
        the object sampling would otherwise have to be re-derived without. Either one
        missing leaves the distance at zero. Both belong to the instrument rather than
        to the scan, which is what makes them registration constants.
        """
        self._detector_pixel_size_m = detector_pixel_size_m
        self._photon_energy_eV = photon_energy_eV

    def get_file_description(self) -> str:
        return 'PEAR Reconstruction Files'

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def _compute_detector_distance_m(
        self, object_pixel_size_m: float, probe_extent: ImageExtent
    ) -> float:
        """Recover the sample-to-detector distance, or zero without the inputs for it.

        The probe supplies the pattern width: it is stored on the detector grid, so
        its array is as wide as the patterns were.
        """
        if self._detector_pixel_size_m is None or self._photon_energy_eV <= 0.0:
            return 0.0

        detector_pixel_geometry = PixelGeometry(
            width_m=self._detector_pixel_size_m, height_m=self._detector_pixel_size_m
        )
        return compute_far_field_propagation_distance(
            detector_pixel_geometry,
            probe_extent,
            wavelength_m=energy_eV_to_wavelength_m(self._photon_energy_eV),
            conjugate_pixel_width_m=object_pixel_size_m,
        )

    def read(self, file_path: Path) -> Product:
        with h5py.File(file_path, 'r') as h5_file:
            # pty-chi holds both in ptychodus order already: the object as
            # (layer, height, width) and the probe as (varying, incoherent, height,
            # width), with the axes present even where there is one of them.
            object_array = h5_file[_OBJECT_ARRAY][()]
            probe_array = h5_file[_PROBE_ARRAY][()]
            pixel_size_m = float(h5_file[_OBJECT_PIXEL_SIZE][()])

            positions = _read_positions(
                h5_file[_POSITIONS_PX][()], object_array.shape[-2:], pixel_size_m
            )

            spacings_dataset = h5_file.get(_SLICE_SPACINGS)
            layer_spacing_m = (
                []
                if spacings_dataset is None
                # Already one per gap, unlike the layer-plus-one list fold_slice keeps.
                else [float(spacing) for spacing in numpy.atleast_1d(spacings_dataset[()])]
            )

            weights_dataset = h5_file.get(_OPR_MODE_WEIGHTS)
            opr_weights = (
                None if weights_dataset is None else numpy.asarray(weights_dataset[()], dtype=float)
            )

            loss_dataset = h5_file.get(_LOSS)
            losses = (
                []
                if loss_dataset is None
                else [
                    LossValue(epoch, float(value))
                    for epoch, value in enumerate(numpy.atleast_1d(loss_dataset[()]))
                ]
            )

        pixel_geometry = PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m)
        metadata = ProductMetadata(
            name=file_path.stem,
            comments='',
            detector_distance_m=self._compute_detector_distance_m(
                pixel_size_m,
                ImageExtent(width_px=probe_array.shape[-1], height_px=probe_array.shape[-2]),
            ),
            photon_energy_eV=self._photon_energy_eV,
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
            mass_attenuation_m2_per_kg=0.0,  # not included in file
            tomography_angle_deg=0.0,  # not included in file
        )

        return Product(
            metadata=metadata,
            probe_positions=positions,
            probes=ProbeEnsemble(
                array=probe_array,
                opr_weights=opr_weights,
                pixel_geometry=pixel_geometry,
            ),
            object_=Object(
                array=object_array,
                pixel_geometry=pixel_geometry,
                # The positions above were converted into the frame centered on the
                # object array, so this states that frame rather than standing in for
                # an unknown: save_product raises on an object with no center.
                center=ObjectCenter(x_m=0.0, y_m=0.0),
                layer_spacing_m=layer_spacing_m,
            ),
            losses=losses,
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.register_product_file_reader_with_adapters(
        # The APS instruments PEAR is driven at read an Eiger at 8 keV. Both are
        # properties of the beamline rather than of a reconstruction, and together
        # they are what turns the recorded sample pixel size into a real distance.
        PEARProductFileReader(detector_pixel_size_m=75e-6, photon_energy_eV=8000.0),
        simple_name=PEARProductFileReader.SIMPLE_NAME,
    )
