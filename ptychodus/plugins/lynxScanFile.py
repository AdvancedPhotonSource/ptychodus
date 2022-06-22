from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Final
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import (ScanDictionary, ScanFileReader, ScanPoint, ScanPointParseError,
                                SimpleScanDictionary)


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
    X_COLUMN: Final[int] = 1
    Y_COLUMN: Final[int] = 2
    DETECTOR_COUNT_COLUMN: Final[int] = 4
    EXPECTED_HEADER: Final[list[str]] = [ 'DataPoint', 'x_st_fzp', 'y_st_fzp', \
            'ckUser_Clk_Count', 'Detector_Count' ]

    @property
    def simpleName(self) -> str:
        return 'Lynx'

    @property
    def fileFilter(self) -> str:
        return 'Lynx Scan Files (*.dat)'

    def read(self, filePath: Path) -> ScanDictionary:
        scanPointDict: dict[int, LynxScanPointList] = defaultdict(LynxScanPointList)

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=' ')
            csvIterator = iter(csvReader)

            metadataRow = next(csvIterator)
            headerRow = next(csvIterator)

            assert headerRow == LynxScanFileReader.EXPECTED_HEADER

            for row in csvIterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(LynxScanFileReader.EXPECTED_HEADER):
                    raise ScanPointParseError()

                detectorCount = int(row[LynxScanFileReader.DETECTOR_COUNT_COLUMN])
                x_um = Decimal(row[LynxScanFileReader.X_COLUMN])
                y_um = Decimal(row[LynxScanFileReader.Y_COLUMN])

                scanPointDict[detectorCount].append(x_um, y_um)

        scanPointList = [points.mean() for _, points in sorted(scanPointDict.items())]

        return SimpleScanDictionary.createFromUnnamedSequence(scanPointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(LynxScanFileReader())
