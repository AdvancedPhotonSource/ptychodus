from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from statistics import median
from typing import Final
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import (ScanDictionary, ScanFileReader, ScanPoint, ScanPointParseError,
                                SimpleScanDictionary)


class LynxScanFileColumn(IntEnum):
    X = 1
    Y = 2
    DETECTOR_COUNT = 4


@dataclass(frozen=True)
class LynxScanPoint:
    x_um: Decimal
    y_um: Decimal


class LynxScanFileReader(ScanFileReader):
    EXPECTED_HEADER: Final[list[str]] = [ 'DataPoint', 'x_st_fzp', 'y_st_fzp', \
            'ckUser_Clk_Count', 'Detector_Count' ]

    @property
    def simpleName(self) -> str:
        return 'Lynx'

    @property
    def fileFilter(self) -> str:
        return 'Lynx Scan Files (*.dat)'

    def read(self, filePath: Path) -> ScanDictionary:
        pointDict: dict[int, list[LynxScanPoint]] = defaultdict(list[LynxScanPoint])
        pointList: list[ScanPoint] = list()

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=' ')
            csvIterator = iter(csvReader)

            metadataRow = next(csvIterator)
            headerRow = next(csvIterator)

            if headerRow != LynxScanFileReader.EXPECTED_HEADER:
                raise ScanPointParseError('Bad header!')

            for row in csvIterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(LynxScanFileReader.EXPECTED_HEADER):
                    raise ScanPointParseError('Bad number of columns!')

                x_um = Decimal(row[LynxScanFileColumn.X])
                y_um = Decimal(row[LynxScanFileColumn.Y])
                detector_count = int(row[LynxScanFileColumn.DETECTOR_COUNT])

                point = LynxScanPoint(x_um, y_um)
                pointDict[detector_count].append(point)

        for _, lynxPointList in pointDict.items():
            xf_um = median([p.x_um for p in lynxPointList])
            yf_um = median([p.y_um for p in lynxPointList])

            um_to_m = Decimal('1e-6')
            x_m = Decimal(xf_um) * um_to_m
            y_m = Decimal(yf_um) * um_to_m

            pointList.append(ScanPoint(x_m, y_m))

        return SimpleScanDictionary({self.simpleName: pointList})
