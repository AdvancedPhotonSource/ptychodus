from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan

logger = logging.getLogger(__name__)


class LYNXOrchestraScanFileReader(ScanFileReader):
    MICRONS_TO_METERS: Final[Decimal] = Decimal('1e-6')
    DATA_POINT_COLUMN: Final[int] = 0
    X_COLUMN: Final[int] = 2
    Y_COLUMN: Final[int] = 5

    EXPECTED_HEADER: Final[list[str]] = [
        'DataPoint',
        'TotalPoints',
        'Target_x',
        'Average_x_st_fzp',
        'Stdev_x_st_fzp',
        'Target_y',
        'Average_y_st_fzp',
        'Stdev_y_st_fzp',
        'Average_cap1',
        'Stdev_cap1',
        'Average_cap2',
        'Stdev_cap2',
        'Average_cap3',
        'Stdev_cap3',
        'Average_cap4',
        'Stdev_cap4',
        'Average_cap5',
        'Stdev_cap5',
    ]

    @property
    def simpleName(self) -> str:
        return 'LYNXOrchestra'

    @property
    def fileFilter(self) -> str:
        return 'LYNX Orchestra Scan Files (*.dat)'

    def read(self, filePath: Path) -> Scan:
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

            if columnHeaderRow == LYNXOrchestraScanFileReader.EXPECTED_HEADER:
                logger.debug(f'Reading scan positions for \"{scanName}\"...')
            else:
                raise ScanPointParseError('Bad header!')

            for row in csvIterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(columnHeaderRow):
                    raise ScanPointParseError('Bad number of columns!')

                data_point = int(row[self.DATA_POINT_COLUMN])
                point = ScanPoint(
                    x=Decimal(row[self.X_COLUMN]) * self.MICRONS_TO_METERS,
                    y=Decimal(row[self.Y_COLUMN]) * self.MICRONS_TO_METERS,
                )
                pointSeqMap[data_point].append(point)

        return TabularScan.createFromMappedPointIterable(pointSeqMap)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(LYNXOrchestraScanFileReader())
