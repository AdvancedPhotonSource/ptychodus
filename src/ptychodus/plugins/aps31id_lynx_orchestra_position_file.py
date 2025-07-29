from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class LYNXOrchestraPositionFileReader(PositionFileReader):
    SIMPLE_NAME: Final[str] = 'APS_LYNX_Orchestra'
    DISPLAY_NAME: Final[str] = 'APS 31-ID-E LYNX Orchestra Files (*.dat)'
    ONE_MICRON_M: Final[float] = 1.0e-6
    DATA_POINT_COLUMN: Final[int] = 0
    X_COLUMN: Final[int] = 3
    Y_COLUMN: Final[int] = 6

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

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()
        scan_name = self.SIMPLE_NAME

        with file_path.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)
            csv_iterator = iter(csv_reader)

            title_row = next(csv_iterator)

            try:
                scan_name = ' '.join(title_row).split(',', maxsplit=1)[0]
            except IndexError:
                raise ScanPointParseError('Bad scan name!')

            column_header_row = next(csv_iterator)

            if column_header_row == LYNXOrchestraPositionFileReader.EXPECTED_HEADER:
                logger.debug(f'Reading scan positions for "{scan_name}"...')
            else:
                raise ScanPointParseError(
                    'Bad LYNX Orchestra header!\n'
                    f'Expected: {LYNXOrchestraPositionFileReader.EXPECTED_HEADER}\n'
                    f'Found:    {column_header_row}\n'
                )

            for row in csv_iterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(column_header_row):
                    raise ScanPointParseError('Bad number of columns!')

                point = ScanPoint(
                    int(row[self.DATA_POINT_COLUMN]),
                    -float(row[self.X_COLUMN]) * self.ONE_MICRON_M,
                    -float(row[self.Y_COLUMN]) * self.ONE_MICRON_M,
                )
                point_list.append(point)

        return PositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        LYNXOrchestraPositionFileReader(),
        simple_name=LYNXOrchestraPositionFileReader.SIMPLE_NAME,
        display_name=LYNXOrchestraPositionFileReader.DISPLAY_NAME,
    )
