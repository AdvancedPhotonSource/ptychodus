from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter, ProductMetadata
from ptychodus.api.scan import Scan, ScanPoint

logger = logging.getLogger(__name__)


class H5ProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'HDF5'
    DISPLAY_NAME: Final[str] = 'Ptychodus Product Files (*.h5 *.hdf5)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    PROBE_ENERGY: Final[str] = 'probe_energy_eV'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'

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
            metadata = ProductMetadata(
                name=h5File.attrs[self.NAME].asstr()[()],
                comments=h5File.attrs[self.COMMENTS].asstr()[()],
                probeEnergyInElectronVolts=float(h5File.attrs[self.PROBE_ENERGY]),
                detectorDistanceInMeters=float(h5File.attrs[self.DETECTOR_OBJECT_DISTANCE]),
            )

            h5ScanIndexes = h5File[self.PROBE_POSITION_INDEXES]
            h5ScanX = h5File[self.PROBE_POSITION_X]
            h5ScanY = h5File[self.PROBE_POSITION_Y]

            for idx, x_m, y_m in zip(h5ScanIndexes[()], h5ScanX[()], h5ScanY[()]):
                point = ScanPoint(idx, x_m, y_m)
                scanPointList.append(point)

            h5Probe = h5File[self.PROBE_ARRAY]
            probe = Probe(
                array=h5Probe[()],
                pixelWidthInMeters=float(h5Probe.attrs[self.PROBE_PIXEL_WIDTH]),
                pixelHeightInMeters=float(h5Probe.attrs[self.PROBE_PIXEL_HEIGHT]),
            )

            h5Object = h5File[self.OBJECT_ARRAY]
            h5ObjectLayerDistance = h5File[self.OBJECT_LAYER_DISTANCE]
            object_ = Object(
                array=h5Object[()],
                layerDistanceInMeters=h5ObjectLayerDistance[()],
                pixelWidthInMeters=float(h5Object[self.OBJECT_PIXEL_WIDTH]),
                pixelHeightInMeters=float(h5Object[self.OBJECT_PIXEL_HEIGHT]),
                centerXInMeters=float(h5Object[self.OBJECT_CENTER_X]),
                centerYInMeters=float(h5Object[self.OBJECT_CENTER_Y]),
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
            scanXInMeters.append(point.positionXInMeters)
            scanYInMeters.append(point.positionYInMeters)

        with h5py.File(filePath, 'w') as h5File:
            metadata = product.metadata
            h5File.attrs[self.NAME] = metadata.name
            h5File.attrs[self.COMMENTS] = metadata.comments
            h5File.attrs[self.DETECTOR_OBJECT_DISTANCE] = metadata.detectorDistanceInMeters
            h5File.attrs[self.PROBE_ENERGY] = metadata.probeEnergyInElectronVolts

            h5File.create_datset(self.PROBE_POSITION_INDEXES, data=scanIndexes)
            h5File.create_dataset(self.PROBE_POSITION_X, data=scanXInMeters)
            h5File.create_dataset(self.PROBE_POSITION_Y, data=scanYInMeters)

            probe = product.probe
            probeGeometry = probe.getGeometry()
            h5Probe = h5File.create_dataset(self.PROBE_ARRAY, data=probe.array)
            h5Probe.attrs[self.PROBE_PIXEL_WIDTH] = probeGeometry.pixelWidthInMeters
            h5Probe.attrs[self.PROBE_PIXEL_HEIGHT] = probeGeometry.pixelHeightInMeters

            object_ = product.object_
            objectGeometry = object_.getGeometry()
            h5Object = h5File.create_dataset(self.OBJECT_ARRAY, data=object_.array)
            h5Object.attrs[self.OBJECT_CENTER_X] = objectGeometry.centerXInMeters
            h5Object.attrs[self.OBJECT_CENTER_Y] = objectGeometry.centerYInMeters
            h5Object.attrs[self.OBJECT_PIXEL_WIDTH] = objectGeometry.pixelWidthInMeters
            h5Object.attrs[self.OBJECT_PIXEL_HEIGHT] = objectGeometry.pixelHeightInMeters
            h5File.create_dataset(self.OBJECT_LAYER_DISTANCE, data=object_.layerDistanceInMeters)

            h5File.create_dataset(self.COSTS_ARRAY, data=product.costs)


def registerPlugins(registry: PluginRegistry) -> None:
    h5ProductFileIO = H5ProductFileIO()

    registry.productFileReaders.registerPlugin(
        h5ProductFileIO,
        simpleName=H5ProductFileIO.SIMPLE_NAME,
        displayName=H5ProductFileIO.DISPLAY_NAME,
    )
    registry.productFileWriters.registerPlugin(
        h5ProductFileIO,
        simpleName=H5ProductFileIO.SIMPLE_NAME,
        displayName=H5ProductFileIO.DISPLAY_NAME,
    )
