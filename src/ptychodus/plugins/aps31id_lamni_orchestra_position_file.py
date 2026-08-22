from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.constants import LengthUnit
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePosition,
    ProbePositionParseError,
)

logger = logging.getLogger(__name__)


class LamNIOrchestraPositionFileReader(ProbePositionFileReader):
    SIMPLE_NAME: Final[str] = 'APS_LamNI_Orchestra'
    DISPLAY_NAME: Final[str] = 'APS 31-ID-E LamNI Orchestra Files (*.dat)'
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

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()
        scan_name = self.SIMPLE_NAME

        with file_path.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)
            csv_iterator = iter(csv_reader)

            title_row = next(csv_iterator)

            try:
                scan_name = ' '.join(title_row).split(',', maxsplit=1)[0]
            except IndexError:
                raise ProbePositionParseError('Bad scan name!')

            column_header_row = next(csv_iterator)

            if column_header_row == LamNIOrchestraPositionFileReader.EXPECTED_HEADER:
                logger.debug(f'Reading scan positions for "{scan_name}"...')
            else:
                raise ProbePositionParseError(
                    'Bad LamNI Orchestra header!\n'
                    f'Expected: {LamNIOrchestraPositionFileReader.EXPECTED_HEADER}\n'
                    f'Found:    {column_header_row}\n'
                )

            for row in csv_iterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(column_header_row):
                    raise ProbePositionParseError('Bad number of columns!')

                point = ProbePosition(
                    int(row[self.DATA_POINT_COLUMN]),
                    -LengthUnit.MICROMETER.to_meters(float(row[self.X_COLUMN])),
                    -LengthUnit.MICROMETER.to_meters(float(row[self.Y_COLUMN])),
                )
                point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        LamNIOrchestraPositionFileReader(),
        simple_name=LamNIOrchestraPositionFileReader.SIMPLE_NAME,
        display_name=LamNIOrchestraPositionFileReader.DISPLAY_NAME,
    )
