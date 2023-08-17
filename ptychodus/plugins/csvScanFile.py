from pathlib import Path
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import (Scan, ScanFileReader, ScanFileWriter, ScanPoint,
                                ScanPointParseError, TabularScan)


class CSVScanFileReader(ScanFileReader):

    def __init__(self, xcol: int = 1, ycol: int = 0) -> None:
        self._xcol = xcol
        self._ycol = ycol

    def read(self, filePath: Path) -> Scan:
        pointList = list()
        minimumColumnCount = max(self._xcol, self._ycol) + 1

        with filePath.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < minimumColumnCount:
                    raise ScanPointParseError('Bad number of columns!')

                point = ScanPoint(
                    x=float(row[self._xcol]),
                    y=float(row[self._ycol]),
                )
                pointList.append(point)

        return TabularScan.createFromPointIterable(pointList)


class CSVScanFileWriter(ScanFileWriter):

    def write(self, filePath: Path, scan: Scan) -> None:
        with filePath.open(mode='wt') as csvFile:
            for index, point in scan.items():
                csvFile.write(f'{point.y},{point.x}\n')


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        CSVScanFileReader(),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
    registry.scanFileWriters.registerPlugin(
        CSVScanFileWriter(),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
