from pathlib import Path
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import (Scan, ScanFileReader, ScanFileWriter, ScanPoint,
                                ScanPointParseError, TabularScan)


class DelimitedScanFileReader(ScanFileReader):

    def __init__(self, delimiter: str, swapXY: bool) -> None:
        self._delimiter = delimiter
        self._swapXY = swapXY

    def read(self, filePath: Path) -> Scan:
        pointList = list()

        if self._swapXY:
            xcol = 1
            ycol = 0
        else:
            xcol = 0
            ycol = 1

        with filePath.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=self._delimiter)

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < 2:
                    raise ScanPointParseError('Bad number of columns!')

                point = ScanPoint(x=float(row[xcol]), y=float(row[ycol]))
                pointList.append(point)

        return TabularScan.createFromPointIterable(pointList)


class DelimitedScanFileWriter(ScanFileWriter):

    def __init__(self, delimiter: str, swapXY: bool) -> None:
        self._delimiter = delimiter
        self._swapXY = swapXY

    def write(self, filePath: Path, scan: Scan) -> None:
        with filePath.open(mode='wt') as csvFile:
            for index, point in scan.items():
                line = f'{point.y}{self._delimiter}{point.x}\n' if self._swapXY \
                        else f'{point.x}{self._delimiter}{point.y}\n'
                csvFile.write(line)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        DelimitedScanFileReader(' ', swapXY=False),
        simpleName='TXT',
        displayName='Space-Separated Values Files (*.txt)',
    )
    registry.scanFileWriters.registerPlugin(
        DelimitedScanFileWriter(' ', swapXY=False),
        simpleName='TXT',
        displayName='Space-Separated Values Files (*.txt)',
    )
    registry.scanFileReaders.registerPlugin(
        DelimitedScanFileReader(',', swapXY=True),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
    registry.scanFileWriters.registerPlugin(
        DelimitedScanFileWriter(',', swapXY=True),
        simpleName='CSV',
        displayName='Comma-Separated Values Files (*.csv)',
    )
