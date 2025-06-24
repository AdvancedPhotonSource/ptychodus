from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class LYNXSoftGlueZynqPositionFileReader(PositionFileReader):
    SIMPLE_NAME: Final[str] = 'APS_LYNX_SoftGlueZynq'
    DISPLAY_NAME: Final[str] = 'APS 31-ID-E LYNX SoftGlueZynq Files (*.dat)'
    ONE_MICRON_M: Final[float] = 1.0e-6

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

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()
        scan_name = self.SIMPLE_NAME

        with file_path.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            csv_iterator = iter(csv_reader)

            title_row = next(csv_iterator)

            try:
                scan_name = ' '.join(title_row).split(',', maxsplit=1)[0]
            except IndexError:
                raise ScanPointParseError('Bad scan name!')

            column_header_row = next(csv_iterator)

            if column_header_row == LYNXSoftGlueZynqPositionFileReader.EXPECTED_HEADER_RAW:
                logger.debug(f'Reading raw scan positions for "{scan_name}"...')
                X = 1  # noqa: N806
                Y = 2  # noqa: N806
                DETECTOR_COUNT = 4  # noqa: N806
            elif column_header_row == LYNXSoftGlueZynqPositionFileReader.EXPECTED_HEADER_PROCESSED:
                logger.debug(f'Reading processed scan positions for "{scan_name}"...')
                DETECTOR_COUNT = 0  # noqa: N806
                X = 1  # noqa: N806
                Y = 3  # noqa: N806
            else:
                raise ScanPointParseError(
                    f'Bad LYNX SoftGlueZynq header!\nFound:    {column_header_row}\n'
                )

            for row in csv_iterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(column_header_row):
                    raise ScanPointParseError('Bad number of columns!')

                point = ScanPoint(
                    int(row[DETECTOR_COUNT]),
                    -float(row[X]) * self.ONE_MICRON_M,
                    -float(row[Y]) * self.ONE_MICRON_M,
                )
                point_list.append(point)

        return PositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        LYNXSoftGlueZynqPositionFileReader(),
        simple_name=LYNXSoftGlueZynqPositionFileReader.SIMPLE_NAME,
        display_name=LYNXSoftGlueZynqPositionFileReader.DISPLAY_NAME,
    )
