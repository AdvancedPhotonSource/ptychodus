from pathlib import Path
from typing import Final, Sequence

import numpy
import scipy.io

from ptychodus.api.constants import wavelength_m_to_energy_eV
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.probe_positions import ProbePositionSequence, ProbePosition
from ptychodus.api.propagate import compute_far_field_propagation_distance
from ptychodus.api.reconstruct import LossValue


class FoldSliceProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'fold_slice_mat'
    DISPLAY_NAME: Final[str] = 'fold_slice Files (*.mat)'

    def __init__(self, detector_pixel_size_m: float | None = None) -> None:
        """Read a fold_slice reconstruction.

        `detector_pixel_size_m` stands in for a pitch the file does not record, and is
        what lets the sample-to-detector distance be recovered from the sample pixel
        size the file does record. Without it the distance reads as zero, which leaves
        the product unusable anywhere the object sampling has to be re-derived.
        """
        self._detector_pixel_size_m = detector_pixel_size_m

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

        mat_dict = scipy.io.loadmat(file_path, simplify_cells=True)
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
        pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)

        probe_array = mat_dict['probe']

        if probe_array.ndim == 3:
            # probe_array[height, width, num_shared_modes]
            probe_array = probe_array.transpose(2, 0, 1)
        elif probe_array.ndim == 4:
            # probe_array[height, width, num_shared_modes, num_varying_modes]
            probe_array = probe_array.transpose(3, 2, 0, 1)

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
        probe_positions = outputs_struct['probe_positions']

        for idx, pos_px in enumerate(probe_positions):
            point = ProbePosition(
                idx,
                pos_px[0] * pixel_width_m,
                pos_px[1] * pixel_height_m,
            )
            point_list.append(point)

        probe = ProbeSequence(
            array=probe_array,
            opr_weights=None,  # TODO OPR, if available
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
            num_spaces = object_array.shape[-3] - 1
            layer_spacing_m = numpy.squeeze(z_distance)[:num_spaces]

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

        fourier_error_out = outputs_struct['fourier_error_out']
        losses: list[LossValue] = []

        for epoch, value in enumerate(fourier_error_out):
            loss = LossValue(epoch, value)
            losses.append(loss)

        return Product(
            metadata=metadata,
            probe_positions=ProbePositionSequence(point_list),
            probes=probe,
            object_=object_,
            losses=losses,
        )
