from pathlib import Path
from typing import Any, Final
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.product import (
    Product,
    ProductFileReader,
    ProductFileWriter,
    ProductMetadata,
)
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

logger = logging.getLogger(__name__)


class NPZProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'Ptychodus NumPy Zipped Archive (*.npz)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'
    PROBE_ENERGY: Final[str] = 'probe_energy_eV'
    PROBE_PHOTON_COUNT: Final[str] = 'probe_photon_count'
    EXPOSURE_TIME: Final[str] = 'exposure_time_s'

    PROBE_ARRAY: Final[str] = 'probe'
    PROBE_PIXEL_HEIGHT: Final[str] = 'probe_pixel_height_m'
    PROBE_PIXEL_WIDTH: Final[str] = 'probe_pixel_width_m'
    PROBE_POSITION_INDEXES: Final[str] = 'probe_position_indexes'
    PROBE_POSITION_X: Final[str] = 'probe_position_x_m'
    PROBE_POSITION_Y: Final[str] = 'probe_position_y_m'

    OBJECT_ARRAY: Final[str] = 'object'
    OBJECT_CENTER_X: Final[str] = 'object_center_x_m'
    OBJECT_CENTER_Y: Final[str] = 'object_center_y_m'
    OBJECT_LAYER_DISTANCE: Final[str] = 'object_layer_distance_m'
    OBJECT_PIXEL_HEIGHT: Final[str] = 'object_pixel_height_m'
    OBJECT_PIXEL_WIDTH: Final[str] = 'object_pixel_width_m'

    COSTS_ARRAY: Final[str] = 'costs'

    def read(self, filePath: Path) -> Product:
        with numpy.load(filePath) as npzFile:
            probePhotonCount = 0.0

            try:
                probePhotonCount = float(npzFile[self.PROBE_PHOTON_COUNT])
            except KeyError:
                logger.debug('Probe photon count not found.')

            metadata = ProductMetadata(
                name=str(npzFile[self.NAME]),
                comments=str(npzFile[self.COMMENTS]),
                detector_distance_m=float(npzFile[self.DETECTOR_OBJECT_DISTANCE]),
                probe_energy_eV=float(npzFile[self.PROBE_ENERGY]),
                probe_photon_count=probePhotonCount,
                exposure_time_s=float(npzFile[self.EXPOSURE_TIME]),
            )

            scanIndexes = npzFile[self.PROBE_POSITION_INDEXES]
            scanXInMeters = npzFile[self.PROBE_POSITION_X]
            scanYInMeters = npzFile[self.PROBE_POSITION_Y]

            probePixelGeometry = PixelGeometry(
                width_m=float(npzFile[self.PROBE_PIXEL_WIDTH]),
                height_m=float(npzFile[self.PROBE_PIXEL_HEIGHT]),
            )
            probe = Probe(array=npzFile[self.PROBE_ARRAY], pixel_geometry=probePixelGeometry)

            objectPixelGeometry = PixelGeometry(
                width_m=float(npzFile[self.OBJECT_PIXEL_WIDTH]),
                height_m=float(npzFile[self.OBJECT_PIXEL_HEIGHT]),
            )
            objectCenter = ObjectCenter(
                position_x_m=float(npzFile[self.OBJECT_CENTER_X]),
                position_y_m=float(npzFile[self.OBJECT_CENTER_Y]),
            )
            object_ = Object(
                array=npzFile[self.OBJECT_ARRAY],
                pixel_geometry=objectPixelGeometry,
                center=objectCenter,
                layer_distance_m=npzFile[self.OBJECT_LAYER_DISTANCE],
            )

            costs = npzFile[self.COSTS_ARRAY]

        scanPointList: list[ScanPoint] = list()

        for idx, x_m, y_m in zip(scanIndexes, scanXInMeters, scanYInMeters):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        return Product(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=probe,
            object_=object_,
            costs=costs,
        )

    def write(self, filePath: Path, product: Product) -> None:
        contents: dict[str, Any] = dict()
        scanIndexes: list[int] = list()
        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for point in product.scan:
            scanIndexes.append(point.index)
            scanXInMeters.append(point.position_x_m)
            scanYInMeters.append(point.position_y_m)

        metadata = product.metadata
        contents[self.NAME] = metadata.name
        contents[self.COMMENTS] = metadata.comments
        contents[self.DETECTOR_OBJECT_DISTANCE] = metadata.detector_distance_m
        contents[self.PROBE_ENERGY] = metadata.probe_energy_eV
        contents[self.PROBE_PHOTON_COUNT] = metadata.probe_photon_count
        contents[self.EXPOSURE_TIME] = metadata.exposure_time_s

        contents[self.PROBE_POSITION_INDEXES] = scanIndexes
        contents[self.PROBE_POSITION_X] = scanXInMeters
        contents[self.PROBE_POSITION_Y] = scanYInMeters

        probe = product.probe
        probeGeometry = probe.get_geometry()
        contents[self.PROBE_ARRAY] = probe.get_array()
        contents[self.PROBE_PIXEL_WIDTH] = probeGeometry.pixel_width_m
        contents[self.PROBE_PIXEL_HEIGHT] = probeGeometry.pixel_height_m

        object_ = product.object_
        objectGeometry = object_.get_geometry()
        contents[self.OBJECT_ARRAY] = object_.get_array()
        contents[self.OBJECT_CENTER_X] = objectGeometry.center_x_m
        contents[self.OBJECT_CENTER_Y] = objectGeometry.center_y_m
        contents[self.OBJECT_PIXEL_WIDTH] = objectGeometry.pixel_width_m
        contents[self.OBJECT_PIXEL_HEIGHT] = objectGeometry.pixel_height_m
        contents[self.OBJECT_LAYER_DISTANCE] = object_.layer_distance_m

        contents[self.COSTS_ARRAY] = product.costs

        numpy.savez(filePath, **contents)


class NPZScanFileReader(ScanFileReader):
    def read(self, filePath: Path) -> Scan:
        with numpy.load(filePath) as npzFile:
            scanIndexes = npzFile[NPZProductFileIO.PROBE_POSITION_INDEXES]
            scanXInMeters = npzFile[NPZProductFileIO.PROBE_POSITION_X]
            scanYInMeters = npzFile[NPZProductFileIO.PROBE_POSITION_Y]

        scanPointList: list[ScanPoint] = list()

        for idx, x_m, y_m in zip(scanIndexes, scanXInMeters, scanYInMeters):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        return Scan(scanPointList)


class NPZProbeFileReader(ProbeFileReader):
    def read(self, filePath: Path) -> Probe:
        with numpy.load(filePath) as npzFile:
            pixelGeometry = PixelGeometry(
                width_m=float(npzFile[NPZProductFileIO.PROBE_PIXEL_WIDTH]),
                height_m=float(npzFile[NPZProductFileIO.PROBE_PIXEL_HEIGHT]),
            )
            return Probe(array=npzFile[NPZProductFileIO.PROBE_ARRAY], pixel_geometry=pixelGeometry)


class NPZObjectFileReader(ObjectFileReader):
    def read(self, filePath: Path) -> Object:
        with numpy.load(filePath) as npzFile:
            pixelGeometry = PixelGeometry(
                width_m=float(npzFile[NPZProductFileIO.OBJECT_PIXEL_WIDTH]),
                height_m=float(npzFile[NPZProductFileIO.OBJECT_PIXEL_HEIGHT]),
            )
            center = ObjectCenter(
                position_x_m=float(npzFile[NPZProductFileIO.OBJECT_CENTER_X]),
                position_y_m=float(npzFile[NPZProductFileIO.OBJECT_CENTER_Y]),
            )
            return Object(
                array=npzFile[NPZProductFileIO.OBJECT_ARRAY],
                pixel_geometry=pixelGeometry,
                center=center,
                layer_distance_m=npzFile[NPZProductFileIO.OBJECT_LAYER_DISTANCE],
            )


def register_plugins(registry: PluginRegistry) -> None:
    npzProductFileIO = NPZProductFileIO()

    registry.product_file_readers.register_plugin(
        npzProductFileIO,
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.product_file_writers.register_plugin(
        npzProductFileIO,
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.scan_file_readers.register_plugin(
        NPZScanFileReader(),
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.probe_file_readers.register_plugin(
        NPZProbeFileReader(),
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.object_file_readers.register_plugin(
        NPZObjectFileReader(),
        simple_name=NPZProductFileIO.SIMPLE_NAME,
        display_name=NPZProductFileIO.DISPLAY_NAME,
    )
