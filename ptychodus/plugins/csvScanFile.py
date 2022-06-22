from decimal import Decimal
from pathlib import Path
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import (ScanDictionary, ScanFileReader, ScanFileWriter, ScanPoint,
                                ScanPointParseError, SimpleScanDictionary)


class CSVScanFileReader(ScanFileReader):

    def __init__(self, xcol: int = 1, ycol: int = 0) -> None:
        self._xcol = xcol
        self._ycol = ycol

    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> ScanDictionary:
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

        return SimpleScanDictionary.createFromUnnamedSequence(scanPointList)


class CSVScanFileWriter(ScanFileWriter):

    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def write(self, filePath: Path, scanDict: ScanDictionary) -> None:
        if len(scanDict) != 1:
            className = type(self).__name__
            raise ValueError(f'{className} only supports single sequence scan files!')

        with open(filePath, 'wt') as csvFile:
            for sequence in scanDict.values():
                for point in sequence:
                    csvFile.write(f'{point.y},{point.x}\n')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(CSVScanFileReader())
    registry.registerPlugin(CSVScanFileWriter())
