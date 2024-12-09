from pathlib import Path
from typing import Final, Sequence

import h5py

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product, ProductFileReader, ProductMetadata
from ptychodus.api.scan import Scan, ScanPoint


class NSLSIIProductFileReader(ProductFileReader):
    SIMPLE_NAME: Final[str] = 'NSLS-II'
    DISPLAY_NAME: Final[str] = 'NSLS-II Product Files (*.mat)'
    ONE_MICRON_M: Final[float] = 1e-6

    def read(self, filePath: Path) -> Product:
        point_list: list[ScanPoint] = list()

        with h5py.File(filePath, 'r') as h5File:
            detector_distance_m = float(h5File['det_dist'][()]) * self.ONE_MICRON_M
            probe_energy_eV = 1000.0 * float(h5File['energy'][()])

            metadata = ProductMetadata(
                name=filePath.stem,
                comments='',
                detectorDistanceInMeters=detector_distance_m,
                probeEnergyInElectronVolts=probe_energy_eV,
                probePhotonCount=0.0,  # not included in file
                exposureTimeInSeconds=0.0,  # not included in file
            )

            pixel_width_m = h5File['img_pixel_size_x'][()]
            pixel_height_m = h5File['img_pixel_size_y'][()]
            pixel_geometry = PixelGeometry(
                widthInMeters=pixel_width_m, heightInMeters=pixel_height_m
            )
            positions_m = h5File['pos_xy'][()].T * self.ONE_MICRON_M

            for index, _xy in enumerate(positions_m):
                point = ScanPoint(
                    index=index,
                    positionXInMeters=_xy[1],
                    positionYInMeters=_xy[2],
                )
                point_list.append(point)

            probe_array = h5File['prb'][()].astype(complex)
            probe = Probe(array=probe_array, pixelGeometry=pixel_geometry)

            object_array = h5File['obj'][()].astype(complex)
            object_ = Object(
                array=object_array,
                pixelGeometry=pixel_geometry,
                center=None,
            )
            costs: Sequence[float] = list()

        return Product(
            metadata=metadata,
            scan=Scan(point_list),
            probe=probe,
            object_=object_,
            costs=costs,
        )


def registerPlugins(registry: PluginRegistry) -> None:
    registry.productFileReaders.registerPlugin(
        NSLSIIProductFileReader(),
        simpleName=NSLSIIProductFileReader.SIMPLE_NAME,
        displayName=NSLSIIProductFileReader.DISPLAY_NAME,
    )
