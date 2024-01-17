from pathlib import Path
from typing import Any, Final

import numpy

from ptychodus.api.experiment import (Experiment, ExperimentFileReader, ExperimentFileWriter,
                                      ExperimentMetadata)
from ptychodus.api.object import Object, ObjectFileReader
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint


class NPZExperimentFileIO(ExperimentFileReader, ExperimentFileWriter):
    SIMPLE_NAME: Final[str] = 'NPZ'
    DISPLAY_NAME: Final[str] = 'NumPy Zipped Archive (*.npz)'

    NAME: Final[str] = 'name'
    COMMENTS: Final[str] = 'comments'
    PROBE_ENERGY: Final[str] = 'probe_energy_eV'
    DETECTOR_OBJECT_DISTANCE: Final[str] = 'detector_object_distance_m'

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

    def read(self, filePath: Path) -> Experiment:
        with numpy.load(filePath) as npzFile:
            metadata = ExperimentMetadata(
                name=str(npzFile[self.NAME]),
                comments=str(npzFile[self.COMMENTS]),
                probeEnergyInElectronVolts=float(npzFile[self.PROBE_ENERGY]),
                detectorObjectDistanceInMeters=float(npzFile[self.DETECTOR_OBJECT_DISTANCE]),
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

        scanPointList: list[ScanPoint] = list()

        for idx, x_m, y_m in zip(scanIndexes, scanXInMeters, scanYInMeters):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        return Experiment(
            metadata=metadata,
            scan=Scan(scanPointList),
            probe=probe,
            object_=object_,
        )

    def write(self, filePath: Path, experiment: Experiment) -> None:
        contents: dict[str, Any] = dict()
        scanIndexes: list[int] = list()
        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for point in experiment.scan:
            scanIndexes.append(point.index)
            scanXInMeters.append(point.positionXInMeters)
            scanYInMeters.append(point.positionYInMeters)

        metadata = experiment.metadata
        contents[self.NAME] = metadata.name
        contents[self.COMMENTS] = metadata.comments
        contents[self.DETECTOR_OBJECT_DISTANCE] = metadata.detectorObjectDistanceInMeters
        contents[self.PROBE_ENERGY] = metadata.probeEnergyInElectronVolts

        contents[self.PROBE_POSITION_INDEXES] = scanIndexes
        contents[self.PROBE_POSITION_X] = scanXInMeters
        contents[self.PROBE_POSITION_Y] = scanYInMeters

        probe = experiment.probe
        contents[self.PROBE_ARRAY] = probe.array
        contents[self.PROBE_PIXEL_WIDTH] = probe.pixelWidthInMeters
        contents[self.PROBE_PIXEL_HEIGHT] = probe.pixelHeightInMeters

        object_ = experiment.object_
        contents[self.OBJECT_ARRAY] = object_.array
        contents[self.OBJECT_CENTER_X] = object_.centerXInMeters
        contents[self.OBJECT_CENTER_Y] = object_.centerYInMeters
        contents[self.OBJECT_PIXEL_WIDTH] = object_.pixelWidthInMeters
        contents[self.OBJECT_PIXEL_HEIGHT] = object_.pixelHeightInMeters
        contents[self.OBJECT_LAYER_DISTANCE] = object_.layerDistanceInMeters

        numpy.savez(filePath, **contents)


class NPZScanFileReader(ScanFileReader):

    def read(self, filePath: Path) -> Scan:
        with numpy.load(filePath) as npzFile:
            scanIndexes = npzFile[NPZExperimentFileIO.PROBE_POSITION_INDEXES]
            scanXInMeters = npzFile[NPZExperimentFileIO.PROBE_POSITION_X]
            scanYInMeters = npzFile[NPZExperimentFileIO.PROBE_POSITION_Y]

        scanPointList: list[ScanPoint] = list()

        for idx, x_m, y_m in zip(scanIndexes, scanXInMeters, scanYInMeters):
            point = ScanPoint(idx, x_m, y_m)
            scanPointList.append(point)

        return Scan(scanPointList)


class NPZProbeFileReader(ProbeFileReader):

    def read(self, filePath: Path) -> Probe:
        with numpy.load(filePath) as npzFile:
            return Probe(
                array=npzFile[NPZExperimentFileIO.PROBE_ARRAY],
                pixelWidthInMeters=float(npzFile[NPZExperimentFileIO.PROBE_PIXEL_WIDTH]),
                pixelHeightInMeters=float(npzFile[NPZExperimentFileIO.PROBE_PIXEL_HEIGHT]),
            )


class NPZObjectFileReader(ObjectFileReader):

    def read(self, filePath: Path) -> Object:
        with numpy.load(filePath) as npzFile:
            return Object(
                array=npzFile[NPZExperimentFileIO.OBJECT_ARRAY],
                layerDistanceInMeters=npzFile[NPZExperimentFileIO.OBJECT_LAYER_DISTANCE],
                pixelWidthInMeters=float(npzFile[NPZExperimentFileIO.OBJECT_PIXEL_WIDTH]),
                pixelHeightInMeters=float(npzFile[NPZExperimentFileIO.OBJECT_PIXEL_HEIGHT]),
                centerXInMeters=float(npzFile[NPZExperimentFileIO.OBJECT_CENTER_X]),
                centerYInMeters=float(npzFile[NPZExperimentFileIO.OBJECT_CENTER_Y]),
            )


def registerPlugins(registry: PluginRegistry) -> None:
    npzExperimentFileIO = NPZExperimentFileIO()

    registry.experimentFileReaders.registerPlugin(
        npzExperimentFileIO,
        simpleName=NPZExperimentFileIO.SIMPLE_NAME,
        displayName=NPZExperimentFileIO.DISPLAY_NAME,
    )
    registry.experimentFileWriters.registerPlugin(
        npzExperimentFileIO,
        simpleName=NPZExperimentFileIO.SIMPLE_NAME,
        displayName=NPZExperimentFileIO.DISPLAY_NAME,
    )
    registry.scanFileReaders.registerPlugin(
        NPZScanFileReader(),
        simpleName=NPZExperimentFileIO.SIMPLE_NAME,
        displayName=NPZExperimentFileIO.DISPLAY_NAME,
    )
    registry.probeFileReaders.registerPlugin(
        NPZProbeFileReader(),
        simpleName=NPZExperimentFileIO.SIMPLE_NAME,
        displayName=NPZExperimentFileIO.DISPLAY_NAME,
    )
    registry.objectFileReaders.registerPlugin(
        NPZObjectFileReader(),
        simpleName=NPZExperimentFileIO.SIMPLE_NAME,
        displayName=NPZExperimentFileIO.DISPLAY_NAME,
    )
