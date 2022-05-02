from decimal import Decimal
from pathlib import Path
from typing import Iterable
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import ScanFileReader, ScanPoint, ScanPointParseError


class CSVScanFileReader(ScanFileReader):
    def __init__(self) -> None:
        self._xcol = 1
        self._ycol = 0

    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        scanPointList = list()
        minimumColumnCount = max(self._xcol, self._ycol) + 1

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < minimumColumnCount:
                    raise ScanPointParseError()

                x = Decimal(row[self._xcol])
                y = Decimal(row[self._ycol])
                point = ScanPoint(x, y)

                scanPointList.append(point)

        return scanPointList


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CSVScanFileReader())
