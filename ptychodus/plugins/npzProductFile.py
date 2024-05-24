from pathlib import Path
from typing import Any, Final

import numpy

from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter, ProductMetadata
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint


class NPZProductFileIO(ProductFileReader, ProductFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'NumPy Zipped Archive (*.npz)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'
    PROBE_ENERGY: Final[str] = 'probe_energy_eV'
    PROBE_PHOTON_FLUX: Final[str] = 'probe_photons_per_s'
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
            metadata = ProductMetadata(
                name=str(npzFile[self.NAME]),
                comments=str(npzFile[self.COMMENTS]),
                detectorDistanceInMeters=float(npzFile[self.DETECTOR_OBJECT_DISTANCE]),
                probeEnergyInElectronVolts=float(npzFile[self.PROBE_ENERGY]),
                probePhotonsPerSecond=float(npzFile[self.PROBE_PHOTON_FLUX]),
                exposureTimeInSeconds=float(npzFile[self.EXPOSURE_TIME]),
            )

            scanIndexes = npzFile[self.PROBE_POSITION_INDEXES]
            scanXInMeters = npzFile[self.PROBE_POSITION_X]
            scanYInMeters = npzFile[self.PROBE_POSITION_Y]

            probe = Probe(
                array=npzFile[self.PROBE_ARRAY],
                pixelWidthInMeters=float(npzFile[self.PROBE_PIXEL_WIDTH]),
                pixelHeightInMeters=float(npzFile[self.PROBE_PIXEL_HEIGHT]),
            )

            object_ = Object(
                array=npzFile[self.OBJECT_ARRAY],
                layerDistanceInMeters=npzFile[self.OBJECT_LAYER_DISTANCE],
                pixelWidthInMeters=float(npzFile[self.OBJECT_PIXEL_WIDTH]),
                pixelHeightInMeters=float(npzFile[self.OBJECT_PIXEL_HEIGHT]),
                centerXInMeters=float(npzFile[self.OBJECT_CENTER_X]),
                centerYInMeters=float(npzFile[self.OBJECT_CENTER_Y]),
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
            scanXInMeters.append(point.positionXInMeters)
            scanYInMeters.append(point.positionYInMeters)

        metadata = product.metadata
        contents[self.NAME] = metadata.name
        contents[self.COMMENTS] = metadata.comments
        contents[self.DETECTOR_OBJECT_DISTANCE] = metadata.detectorDistanceInMeters
        contents[self.PROBE_ENERGY] = metadata.probeEnergyInElectronVolts
        contents[self.PROBE_PHOTON_FLUX] = metadata.probePhotonsPerSecond
        contents[self.EXPOSURE_TIME] = metadata.exposureTimeInSeconds

        contents[self.PROBE_POSITION_INDEXES] = scanIndexes
        contents[self.PROBE_POSITION_X] = scanXInMeters
        contents[self.PROBE_POSITION_Y] = scanYInMeters

        probe = product.probe
        probeGeometry = probe.getGeometry()
        contents[self.PROBE_ARRAY] = probe.array
        contents[self.PROBE_PIXEL_WIDTH] = probeGeometry.pixelWidthInMeters
        contents[self.PROBE_PIXEL_HEIGHT] = probeGeometry.pixelHeightInMeters

        object_ = product.object_
        objectGeometry = object_.getGeometry()
        contents[self.OBJECT_ARRAY] = object_.array
        contents[self.OBJECT_CENTER_X] = objectGeometry.centerXInMeters
        contents[self.OBJECT_CENTER_Y] = objectGeometry.centerYInMeters
        contents[self.OBJECT_PIXEL_WIDTH] = objectGeometry.pixelWidthInMeters
        contents[self.OBJECT_PIXEL_HEIGHT] = objectGeometry.pixelHeightInMeters
        contents[self.OBJECT_LAYER_DISTANCE] = object_.layerDistanceInMeters

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
            return Probe(
                array=npzFile[NPZProductFileIO.PROBE_ARRAY],
                pixelWidthInMeters=float(npzFile[NPZProductFileIO.PROBE_PIXEL_WIDTH]),
                pixelHeightInMeters=float(npzFile[NPZProductFileIO.PROBE_PIXEL_HEIGHT]),
            )


class NPZObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> Object:
        with numpy.load(filePath) as npzFile:
            return Object(
                array=npzFile[NPZProductFileIO.OBJECT_ARRAY],
                layerDistanceInMeters=npzFile[NPZProductFileIO.OBJECT_LAYER_DISTANCE],
                pixelWidthInMeters=float(npzFile[NPZProductFileIO.OBJECT_PIXEL_WIDTH]),
                pixelHeightInMeters=float(npzFile[NPZProductFileIO.OBJECT_PIXEL_HEIGHT]),
                centerXInMeters=float(npzFile[NPZProductFileIO.OBJECT_CENTER_X]),
                centerYInMeters=float(npzFile[NPZProductFileIO.OBJECT_CENTER_Y]),
            )


def registerPlugins(registry: PluginRegistry) -> None:
    npzProductFileIO = NPZProductFileIO()

    registry.productFileReaders.registerPlugin(
        npzProductFileIO,
        simpleName=NPZProductFileIO.SIMPLE_NAME,
        displayName=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.productFileWriters.registerPlugin(
        npzProductFileIO,
        simpleName=NPZProductFileIO.SIMPLE_NAME,
        displayName=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.scanFileReaders.registerPlugin(
        NPZScanFileReader(),
        simpleName=NPZProductFileIO.SIMPLE_NAME,
        displayName=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.probeFileReaders.registerPlugin(
        NPZProbeFileReader(),
        simpleName=NPZProductFileIO.SIMPLE_NAME,
        displayName=NPZProductFileIO.DISPLAY_NAME,
    )
    registry.objectFileReaders.registerPlugin(
        NPZObjectFileReader(),
        simpleName=NPZProductFileIO.SIMPLE_NAME,
        displayName=NPZProductFileIO.DISPLAY_NAME,
    )
