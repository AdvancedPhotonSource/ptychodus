from collections import defaultdict
from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan

logger = logging.getLogger(__name__)


class LYNXOrchestraScanFileReader(ScanFileReader):
    SIMPLE_NAME: Final[str] = 'LYNXOrchestra'
    MICRONS_TO_METERS: Final[float] = 1.e-6
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

    def read(self, filePath: Path) -> Scan:
        pointSeqMap: dict[int, list[ScanPoint]] = defaultdict(list[ScanPoint])
        scanName = self.SIMPLE_NAME

        with filePath.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=' ', skipinitialspace=True)
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
                raise ScanPointParseError(
                    'Bad LYNX Orchestra header!\n'
                    f'Expected: {LYNXOrchestraScanFileReader.EXPECTED_HEADER}\n'
                    f'Found:    {columnHeaderRow}\n')

            for row in csvIterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(columnHeaderRow):
                    raise ScanPointParseError('Bad number of columns!')

                data_point = int(row[self.DATA_POINT_COLUMN])
                point = ScanPoint(
                    x=float(row[self.X_COLUMN]) * self.MICRONS_TO_METERS,
                    y=float(row[self.Y_COLUMN]) * self.MICRONS_TO_METERS,
                )
                pointSeqMap[data_point].append(point)

        return TabularScan.createFromMappedPointIterable(pointSeqMap)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        LYNXOrchestraScanFileReader(),
        simpleName=LYNXOrchestraScanFileReader.SIMPLE_NAME,
        displayName='LYNX Orchestra Scan Files (*.dat)',
    )
