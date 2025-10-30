from pathlib import Path
from typing import Final, Sequence

import numpy
import scipy.io

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
    Product,
    ProductFileReader,
    ProductMetadata,
)
from ptychodus.api.probe_positions import ProbePositionSequence, ProbePosition
from ptychodus.api.reconstructor import LossValue


class FoldSliceProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'fold_slice'
    DISPLAY_NAME: Final[str] = 'fold_slice Files (*.mat)'

    def read(self, file_path: Path) -> Product:
        point_list: list[ProbePosition] = list()

        hc_eVm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S / ELECTRON_VOLT_J  # noqa: N806
        mat_dict = scipy.io.loadmat(file_path, simplify_cells=True)
        p_struct = mat_dict['p']
        probe_energy_eV = hc_eVm / p_struct['lambda']  # noqa: N806

        metadata = ProductMetadata(
            name=file_path.stem,
            comments='',
            detector_distance_m=0.0,  # not included in file
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=0.0,  # not included in file
            exposure_time_s=0.0,  # not included in file
            mass_attenuation_m2_kg=0.0,  # not included in file
            tomography_angle_deg=0.0,  # not included in file
        )

        dx_spec = p_struct['dx_spec']
        pixel_width_m = dx_spec[0]
        pixel_height_m = dx_spec[1]
        pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)

        outputs_struct = mat_dict['outputs']
        probe_positions = outputs_struct['probe_positions']

        for idx, pos_px in enumerate(probe_positions):
            point = ProbePosition(
                idx,
                pos_px[0] * pixel_width_m,
                pos_px[1] * pixel_height_m,
            )
            point_list.append(point)

        probe_array = mat_dict['probe']

        if probe_array.ndim == 3:
            # probe_array[height, width, num_shared_modes]
            probe_array = probe_array.transpose(2, 0, 1)
        elif probe_array.ndim == 4:
            # probe_array[height, width, num_shared_modes, num_varying_modes]
            probe_array = probe_array.transpose(3, 2, 0, 1)

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
            center=None,
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


def register_plugins(registry: PluginRegistry) -> None:
    registry.register_product_file_reader_with_adapters(
        FoldSliceProductFileReader(),
        simple_name=FoldSliceProductFileReader.SIMPLE_NAME,
        display_name=FoldSliceProductFileReader.DISPLAY_NAME,
    )
