from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan
from ptychodus.api.state import StateDataRegistry


class NPZScanFileReader(ScanFileReader):

    def read(self, filePath: Path) -> Scan:
        pointMap: dict[int, ScanPoint] = dict()

        npz = numpy.load(filePath)
        positionIndexes = npz[StateDataRegistry.POSITION_INDEXES]
        positionXInMeters = npz[StateDataRegistry.POSITION_X]
        positionYInMeters = npz[StateDataRegistry.POSITION_Y]

        for index, x, y in zip(positionIndexes, positionXInMeters, positionYInMeters):
            pointMap[index] = ScanPoint(x, y)

        return TabularScan(pointMap)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        NPZScanFileReader(),
        simpleName='NPZ',
        displayName=StateDataRegistry.FILE_FILTER,
    )
