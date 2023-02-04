from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan

logger = logging.getLogger(__name__)


class LYNXSoftGlueZynqScanFileReader(ScanFileReader):
    MICRONS_TO_METERS: Final[Decimal] = Decimal('1e-6')

    EXPECTED_HEADER_RAW: Final[list[str]] = [
        'DataPoint',
        'x_st_fzp',
        'y_st_fzp',
        'ckUser_Clk_Count',
        'Detector_Count',
    ]

    EXPECTED_HEADER_PROCESSED: Final[list[str]] = [
        'Detector_Count',
        'Average_x_st_fzp',
        'Stdev_x_st_fzp',
        'Average_y_st_fzp',
        'Stdev_y_st_fzp',
    ]

    @property
    def simpleName(self) -> str:
        return 'LYNXSoftGlueZynq'

    @property
    def fileFilter(self) -> str:
        return 'LYNX SoftGlueZynq Scan Files (*.dat)'

    def read(self, filePath: Path) -> Sequence[Scan]:
        pointSeqMap: dict[int, list[ScanPoint]] = defaultdict(list[ScanPoint])
        scanName = self.simpleName

        with filePath.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=' ')
            csvIterator = iter(csvReader)

            titleRow = next(csvIterator)

            try:
                scanName = ' '.join(titleRow).split(',', maxsplit=1)[0]
            except IndexError:
                raise ScanPointParseError('Bad scan name!')

            columnHeaderRow = next(csvIterator)

            if columnHeaderRow == LYNXSoftGlueZynqScanFileReader.EXPECTED_HEADER_RAW:
                logger.debug(f'Reading raw scan positions from \"{scanName}\"...')
                X = 1
                Y = 2
                DETECTOR_COUNT = 4
            elif columnHeaderRow == LYNXSoftGlueZynqScanFileReader.EXPECTED_HEADER_PROCESSED:
                logger.debug(f'Reading processed scan positions from \"{scanName}\"...')
                DETECTOR_COUNT = 0
                X = 1
                Y = 3
            else:
                raise ScanPointParseError('Bad header!')

            for row in csvIterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(columnHeaderRow):
                    raise ScanPointParseError('Bad number of columns!')

                detector_count = int(row[DETECTOR_COUNT])
                point = ScanPoint(
                    x=Decimal(row[X]) * LYNXSoftGlueZynqScanFileReader.MICRONS_TO_METERS,
                    y=Decimal(row[Y]) * LYNXSoftGlueZynqScanFileReader.MICRONS_TO_METERS,
                )
                pointSeqMap[detector_count].append(point)

        return [TabularScan.createFromMappedPointSequence(scanName, pointSeqMap)]


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(LYNXSoftGlueZynqScanFileReader())
