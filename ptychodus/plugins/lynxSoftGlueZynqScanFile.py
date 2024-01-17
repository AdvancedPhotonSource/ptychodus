from pathlib import Path
from typing import Final
import csv
import logging

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError

logger = logging.getLogger(__name__)


class LYNXSoftGlueZynqScanFileReader(ScanFileReader):
    SIMPLE_NAME: Final[str] = 'LYNXSoftGlueZynq'
    MICRONS_TO_METERS: Final[float] = 1.e-6

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

    def read(self, filePath: Path) -> Scan:
        pointList: list[ScanPoint] = list()
        scanName = self.SIMPLE_NAME

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
                logger.debug(f'Reading raw scan positions for \"{scanName}\"...')
                X = 1
                Y = 2
                DETECTOR_COUNT = 4
            elif columnHeaderRow == LYNXSoftGlueZynqScanFileReader.EXPECTED_HEADER_PROCESSED:
                logger.debug(f'Reading processed scan positions for \"{scanName}\"...')
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

                point = ScanPoint(
                    int(row[DETECTOR_COUNT]),
                    float(row[X]) * LYNXSoftGlueZynqScanFileReader.MICRONS_TO_METERS,
                    float(row[Y]) * LYNXSoftGlueZynqScanFileReader.MICRONS_TO_METERS,
                )
                pointList.append(point)

        return Scan(pointList)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scanFileReaders.registerPlugin(
        LYNXSoftGlueZynqScanFileReader(),
        simpleName=LYNXSoftGlueZynqScanFileReader.SIMPLE_NAME,
        displayName='LYNX SoftGlueZynq Scan Files (*.dat)',
    )
