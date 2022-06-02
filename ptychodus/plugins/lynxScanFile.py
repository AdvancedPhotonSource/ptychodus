from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import ScanFileReader, ScanPoint, ScanPointParseError


class LynxScanPointList:

    def __init__(self) -> None:
        self.xInMicrons: list[Decimal] = list()
        self.yInMicrons: list[Decimal] = list()

    def append(self, x_um: Decimal, y_um: Decimal) -> None:
        self.xInMicrons.append(x_um)
        self.yInMicrons.append(y_um)

    def mean(self) -> ScanPoint:
        micronsToMeters = Decimal('1e-6')
        x_um = Decimal(sum(self.xInMicrons)) / Decimal(len(self.xInMicrons))
        y_um = Decimal(sum(self.yInMicrons)) / Decimal(len(self.yInMicrons))
        return ScanPoint(x_um * micronsToMeters, y_um * micronsToMeters)


class LynxScanFileReader(ScanFileReader):
    X_COLUMN = 1
    Y_COLUMN = 2
    DETECTOR_COUNT_COLUMN = 4

    @property
    def simpleName(self) -> str:
        return f'Lynx'

    @property
    def fileFilter(self) -> str:
        return f'Lynx Scan Files (*.dat)'

    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        scanPointDict: dict[int, LynxScanPointList] = defaultdict(LynxScanPointList)

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=' ')
            csvIterator = iter(csvReader)

            metadataRow = next(csvIterator)
            headerRow = next(csvIterator)

            assert len(headerRow) == 5
            assert headerRow[0] == 'DataPoint'
            assert headerRow[1] == 'x_st_fzp'
            assert headerRow[2] == 'y_st_fzp'
            assert headerRow[3] == 'ckUser_Clk_Count'
            assert headerRow[4] == 'Detector_Count'

            for row in csvIterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != 5:
                    raise ScanPointParseError()

                detectorCount = int(row[LynxScanFileReader.DETECTOR_COUNT_COLUMN])
                x_um = Decimal(row[LynxScanFileReader.X_COLUMN])
                y_um = Decimal(row[LynxScanFileReader.Y_COLUMN])

                scanPointDict[detectorCount].append(x_um, y_um)

        scanPointList = [
            scanPointList.mean() for _, scanPointList in sorted(scanPointDict.items())
        ]
        xMeanInMeters = Decimal(sum(point.x for point in scanPointList)) / len(scanPointList)
        yMeanInMeters = Decimal(sum(point.y for point in scanPointList)) / len(scanPointList)

        for idx, scanPoint in enumerate(scanPointList):
            x_m = scanPoint.x - xMeanInMeters
            y_m = scanPoint.y - yMeanInMeters
            scanPointList[idx] = ScanPoint(x_m, y_m)

        return scanPointList


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(LynxScanFileReader())
