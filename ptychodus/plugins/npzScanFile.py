from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan
from ptychodus.api.state import StateDataRegistry


class NPZScanFileReader(ScanFileReader):

    def read(self, filePath: Path) -> Scan:
        pointMap: dict[int, ScanPoint] = dict()

        npz = numpy.load(filePath)
        scanIndexes = npz[StateDataRegistry.SCAN_INDEXES]
        scanXInMeters = npz[StateDataRegistry.SCAN_X]
        scanYInMeters = npz[StateDataRegistry.SCAN_Y]

        for index, x, y in zip(scanIndexes, scanXInMeters, scanYInMeters):
            pointMap[index] = ScanPoint(x, y)

        return TabularScan(pointMap)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        NPZScanFileReader(),
        simpleName='NPZ',
        displayName=StateDataRegistry.FILE_FILTER,
    )
