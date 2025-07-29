from pathlib import Path
from typing import Final, Sequence

import h5py

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.product import LossValue, Product, ProductFileReader, ProductMetadata
from ptychodus.api.scan import PositionSequence, ScanPoint


class NSLSIIProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'NSLS_II'
    DISPLAY_NAME: Final[str] = 'NSLS-II MATLAB Files (*.mat)'
    ONE_MICRON_M: Final[float] = 1.0e-6

    def read(self, file_path: Path) -> Product:
        point_list: list[ScanPoint] = list()

        with h5py.File(file_path, 'r') as h5_file:
            detector_distance_m = float(h5_file['det_dist'][()]) * self.ONE_MICRON_M
            probe_energy_eV = 1000 * float(h5_file['energy'][()])  # noqa: N806

            metadata = ProductMetadata(
                name=file_path.stem,
                comments='',
                detector_distance_m=detector_distance_m,
                probe_energy_eV=probe_energy_eV,
                probe_photon_count=0.0,  # not included in file
                exposure_time_s=0.0,  # not included in file
                mass_attenuation_m2_kg=0.0,  # not included in file
                tomography_angle_deg=0.0,  # not included in file
            )

            pixel_width_m = h5_file['img_pixel_size_x'][()]
            pixel_height_m = h5_file['img_pixel_size_y'][()]
            pixel_geometry = PixelGeometry(width_m=pixel_width_m, height_m=pixel_height_m)
            positions_m = h5_file['pos_xy'][()].T * self.ONE_MICRON_M

            for index, _xy in enumerate(positions_m):
                point = ScanPoint(
                    index=index,
                    position_x_m=_xy[1],
                    position_y_m=_xy[2],
                )
                point_list.append(point)

            probe_array = h5_file['prb'][()].astype(complex)
            probes = ProbeSequence(
                array=probe_array, opr_weights=None, pixel_geometry=pixel_geometry
            )

            object_array = h5_file['obj'][()].astype(complex)
            object_ = Object(
                array=object_array,
                pixel_geometry=pixel_geometry,
                center=None,
            )
            loss: Sequence[LossValue] = list()

        return Product(
            metadata=metadata,
            positions=PositionSequence(point_list),
            probes=probes,
            object_=object_,
            losses=loss,
        )


def register_plugins(registry: PluginRegistry) -> None:
    registry.register_product_file_reader_with_adapters(
        NSLSIIProductFileReader(),
        simple_name=NSLSIIProductFileReader.SIMPLE_NAME,
        display_name=NSLSIIProductFileReader.DISPLAY_NAME,
    )
