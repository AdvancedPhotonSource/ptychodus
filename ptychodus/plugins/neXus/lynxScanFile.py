from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from typing import Final
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan


class LynxScanFileColumn(IntEnum):
    X = 1
    Y = 2
    DETECTOR_COUNT = 4


class LynxScanFileReader(ScanFileReader):
    EXPECTED_HEADER: Final[list[str]] = [
        'DataPoint',
        'x_st_fzp',
        'y_st_fzp',
        'ckUser_Clk_Count',
        'Detector_Count',
    ]

    @property
    def simpleName(self) -> str:
        return 'LYNX'

    @property
    def fileFilter(self) -> str:
        return 'LYNX Scan Files (*.dat)'

    def read(self, filePath: Path) -> Sequence[Scan]:
        pointSeqMap: dict[int, list[ScanPoint]] = defaultdict(list[ScanPoint])
        micronsToMeters = Decimal('1e-6')

        with filePath.open(newline='') as csvFile:
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

                detector_count = int(row[LynxScanFileColumn.DETECTOR_COUNT])
                point = ScanPoint(
                    x=Decimal(row[LynxScanFileColumn.X]) * micronsToMeters,
                    y=Decimal(row[LynxScanFileColumn.Y]) * micronsToMeters,
                )
                pointSeqMap[detector_count].append(point)

        return [TabularScan.createFromMappedPointSequence(self.simpleName, pointSeqMap)]
