from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.artifact import (Artifact, ArtifactFileReader, ArtifactFileWriter,
                                    ArtifactMetadata)
from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe
from ptychodus.api.scan import Scan, ScanPoint

logger = logging.getLogger(__name__)


class H5ArtifactFileIO(ArtifactFileReader, ArtifactFileWriter):
    SIMPLE_NAME: Final[str] = 'HDF5'
    DISPLAY_NAME: Final[str] = 'Ptychodus Artifact Files (*.h5 *.hdf5)'

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

    def read(self, filePath: Path) -> Artifact:
        scanPointList: list[ScanPoint] = list()

        with h5py.File(filePath, 'r') as h5File:
            metadata = ArtifactMetadata(
                name=h5File.attrs[self.NAME].asstr()[()],
                comments=h5File.attrs[self.COMMENTS].asstr()[()],
                probeEnergyInElectronVolts=float(h5File.attrs[self.PROBE_ENERGY]),
                detectorObjectDistanceInMeters=float(h5File.attrs[self.DETECTOR_OBJECT_DISTANCE]),
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

        return Artifact(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=probe,
            object_=object_,
        )

    def write(self, filePath: Path, artifact: Artifact) -> None:
        scanIndexes: list[int] = list()
        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for point in artifact.scan:
            scanIndexes.append(point.index)
            scanXInMeters.append(point.positionXInMeters)
            scanYInMeters.append(point.positionYInMeters)

        with h5py.File(filePath, 'w') as h5File:
            metadata = artifact.metadata
            h5File.attrs[self.NAME] = metadata.name
            h5File.attrs[self.COMMENTS] = metadata.comments
            h5File.attrs[self.DETECTOR_OBJECT_DISTANCE] = metadata.detectorObjectDistanceInMeters
            h5File.attrs[self.PROBE_ENERGY] = metadata.probeEnergyInElectronVolts

            h5File.create_datset(self.PROBE_POSITION_INDEXES, data=scanIndexes)
            h5File.create_dataset(self.PROBE_POSITION_X, data=scanXInMeters)
            h5File.create_dataset(self.PROBE_POSITION_Y, data=scanYInMeters)

            probe = artifact.probe
            probeGeometry = probe.getGeometry()
            h5Probe = h5File.create_dataset(self.PROBE_ARRAY, data=probe.array)
            h5Probe.attrs[self.PROBE_PIXEL_WIDTH] = probeGeometry.pixelWidthInMeters
            h5Probe.attrs[self.PROBE_PIXEL_HEIGHT] = probeGeometry.pixelHeightInMeters

            object_ = artifact.object_
            objectGeometry = object_.getGeometry()
            h5Object = h5File.create_dataset(self.OBJECT_ARRAY, data=object_.array)
            h5Object.attrs[self.OBJECT_CENTER_X] = objectGeometry.centerXInMeters
            h5Object.attrs[self.OBJECT_CENTER_Y] = objectGeometry.centerYInMeters
            h5Object.attrs[self.OBJECT_PIXEL_WIDTH] = objectGeometry.pixelWidthInMeters
            h5Object.attrs[self.OBJECT_PIXEL_HEIGHT] = objectGeometry.pixelHeightInMeters
            h5File.create_dataset(self.OBJECT_LAYER_DISTANCE, data=object_.layerDistanceInMeters)


def registerPlugins(registry: PluginRegistry) -> None:
    h5ArtifactFileIO = H5ArtifactFileIO()

    registry.artifactFileReaders.registerPlugin(
        h5ArtifactFileIO,
        simpleName=H5ArtifactFileIO.SIMPLE_NAME,
        displayName=H5ArtifactFileIO.DISPLAY_NAME,
    )
    registry.artifactFileWriters.registerPlugin(
        h5ArtifactFileIO,
        simpleName=H5ArtifactFileIO.SIMPLE_NAME,
        displayName=H5ArtifactFileIO.DISPLAY_NAME,
    )
