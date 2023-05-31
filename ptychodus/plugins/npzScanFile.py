from decimal import Decimal
from pathlib import Path

import numpy

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, TabularScan


class NPZScanFileReader(ScanFileReader):

    @property
    def simpleName(self) -> str:
        return 'NPZ'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def read(self, filePath: Path) -> Scan:
        pointMap: dict[int, ScanPoint] = dict()

        npz = numpy.load(filePath)
        scanIndex = npz['scanIndex']
        scanXInMeters = npz['scanXInMeters']
        scanYInMeters = npz['scanYInMeters']

        for index, x, y in zip(scanIndex, scanXInMeters, scanYInMeters):
            pointMap[index] = ScanPoint(
                x=Decimal.from_float(x),
                y=Decimal.from_float(y),
            )

        return TabularScan(pointMap)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(NPZScanFileReader())
