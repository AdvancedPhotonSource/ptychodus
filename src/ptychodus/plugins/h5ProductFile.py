from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe
from ptychodus.api.product import (
    Product,
    ProductFileReader,
    ProductFileWriter,
    ProductMetadata,
)
from ptychodus.api.scan import Scan, ScanPoint

logger = logging.getLogger(__name__)


class H5ProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'HDF5'
    DISPLAY_NAME: Final[str] = 'Ptychodus Product Files (*.h5 *.hdf5)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'
    PROBE_ENERGY: Final[str] = 'probe_energy_eV'
    PROBE_PHOTON_COUNT: Final[str] = 'probe_photon_count'
    EXPOSURE_TIME: Final[str] = 'exposure_time_s'

    PROBE_ARRAY: Final[str] = 'probe'
    PROBE_PIXEL_HEIGHT: Final[str] = 'pixel_height_m'
    PROBE_PIXEL_WIDTH: Final[str] = 'pixel_width_m'
    PROBE_POSITION_INDEXES: Final[str] = 'probe_position_indexes'
    PROBE_POSITION_X: Final[str] = 'probe_position_x_m'
    PROBE_POSITION_Y: Final[str] = 'probe_position_y_m'

    OBJECT_ARRAY: Final[str] = 'object'
    OBJECT_CENTER_X: Final[str] = 'center_x_m'
    OBJECT_CENTER_Y: Final[str] = 'center_y_m'
    OBJECT_LAYER_DISTANCE: Final[str] = 'object_layer_distance_m'
    OBJECT_PIXEL_HEIGHT: Final[str] = 'pixel_height_m'
    OBJECT_PIXEL_WIDTH: Final[str] = 'pixel_width_m'

    COSTS_ARRAY: Final[str] = 'costs'

    def read(self, filePath: Path) -> Product:
        scanPointList: list[ScanPoint] = list()

        with h5py.File(filePath, 'r') as h5File:
            probePhotonCount = 0.0

            try:
                probePhotonCount = float(h5File.attrs[self.PROBE_PHOTON_COUNT])
            except KeyError:
                logger.debug('Probe photon count not found.')

            metadata = ProductMetadata(
                name=str(h5File.attrs[self.NAME]),
                comments=str(h5File.attrs[self.COMMENTS]),
                detector_distance_m=float(h5File.attrs[self.DETECTOR_OBJECT_DISTANCE]),
                probe_energy_eV=float(h5File.attrs[self.PROBE_ENERGY]),
                probe_photon_count=probePhotonCount,
                exposure_time_s=float(h5File.attrs[self.EXPOSURE_TIME]),
            )

            h5ScanIndexes = h5File[self.PROBE_POSITION_INDEXES]
            h5ScanX = h5File[self.PROBE_POSITION_X]
            h5ScanY = h5File[self.PROBE_POSITION_Y]

            for idx, x_m, y_m in zip(h5ScanIndexes[()], h5ScanX[()], h5ScanY[()]):
                point = ScanPoint(idx, x_m, y_m)
                scanPointList.append(point)

            h5Probe = h5File[self.PROBE_ARRAY]
            probePixelGeometry = PixelGeometry(
                width_m=float(h5Probe.attrs[self.PROBE_PIXEL_WIDTH]),
                height_m=float(h5Probe.attrs[self.PROBE_PIXEL_HEIGHT]),
            )
            probe = Probe(
                array=h5Probe[()],
                pixel_geometry=probePixelGeometry,
            )

            h5Object = h5File[self.OBJECT_ARRAY]
            objectPixelGeometry = PixelGeometry(
                width_m=float(h5Object.attrs[self.OBJECT_PIXEL_WIDTH]),
                height_m=float(h5Object.attrs[self.OBJECT_PIXEL_HEIGHT]),
            )
            objectCenter = ObjectCenter(
                position_x_m=float(h5Object.attrs[self.OBJECT_CENTER_X]),
                position_y_m=float(h5Object.attrs[self.OBJECT_CENTER_Y]),
            )
            h5ObjectLayerDistance = h5File[self.OBJECT_LAYER_DISTANCE]
            object_ = Object(
                array=h5Object[()],
                pixel_geometry=objectPixelGeometry,
                center=objectCenter,
                layer_distance_m=h5ObjectLayerDistance[()],
            )

            h5Costs = h5File[self.COSTS_ARRAY]
            costs = h5Costs[()]

        return Product(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=probe,
            object_=object_,
            costs=costs,
        )

    def write(self, filePath: Path, product: Product) -> None:
        scanIndexes: list[int] = list()
        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for point in product.scan:
            scanIndexes.append(point.index)
            scanXInMeters.append(point.position_x_m)
            scanYInMeters.append(point.position_y_m)

        with h5py.File(filePath, 'w') as h5File:
            metadata = product.metadata
            h5File.attrs[self.NAME] = metadata.name
            h5File.attrs[self.COMMENTS] = metadata.comments
            h5File.attrs[self.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
            h5File.attrs[self.PROBE_ENERGY] = metadata.probe_energy_eV
            h5File.attrs[self.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
            h5File.attrs[self.EXPOSURE_TIME] = metadata.exposure_time_s

            h5File.create_dataset(self.PROBE_POSITION_INDEXES, data=scanIndexes)
            h5File.create_dataset(self.PROBE_POSITION_X, data=scanXInMeters)
            h5File.create_dataset(self.PROBE_POSITION_Y, data=scanYInMeters)

            probe = product.probe
            probeGeometry = probe.get_geometry()
            h5Probe = h5File.create_dataset(self.PROBE_ARRAY, data=probe.get_array())
            h5Probe.attrs[self.PROBE_PIXEL_WIDTH] = probeGeometry.pixel_width_m
            h5Probe.attrs[self.PROBE_PIXEL_HEIGHT] = probeGeometry.pixel_height_m

            object_ = product.object_
            objectGeometry = object_.get_geometry()
            h5Object = h5File.create_dataset(self.OBJECT_ARRAY, data=object_.get_array())
            h5Object.attrs[self.OBJECT_CENTER_X] = objectGeometry.center_x_m
            h5Object.attrs[self.OBJECT_CENTER_Y] = objectGeometry.center_y_m
            h5Object.attrs[self.OBJECT_PIXEL_WIDTH] = objectGeometry.pixel_width_m
            h5Object.attrs[self.OBJECT_PIXEL_HEIGHT] = objectGeometry.pixel_height_m
            h5File.create_dataset(self.OBJECT_LAYER_DISTANCE, data=object_.layer_distance_m)

            h5File.create_dataset(self.COSTS_ARRAY, data=product.costs)


def register_plugins(registry: PluginRegistry) -> None:
    h5ProductFileIO = H5ProductFileIO()

    registry.productFileReaders.register_plugin(
        h5ProductFileIO,
        simple_name=H5ProductFileIO.SIMPLE_NAME,
        display_name=H5ProductFileIO.DISPLAY_NAME,
    )
    registry.productFileWriters.register_plugin(
        h5ProductFileIO,
        simple_name=H5ProductFileIO.SIMPLE_NAME,
        display_name=H5ProductFileIO.DISPLAY_NAME,
    )
