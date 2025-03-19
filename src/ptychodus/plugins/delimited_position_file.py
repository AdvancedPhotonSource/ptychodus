from pathlib import Path
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.scan import (
    PositionSequence,
    PositionFileReader,
    PositionFileWriter,
    ScanPoint,
    ScanPointParseError,
)


class DelimitedPositionFileReader(PositionFileReader):
    def __init__(self, delimiter: str, swapXY: bool) -> None:
        self._delimiter = delimiter
        self._swapXY = swapXY

    def read(self, file_path: Path) -> PositionSequence:
        pointList: list[ScanPoint] = list()

        if self._swapXY:
            xcol = 1
            ycol = 0
        else:
            xcol = 0
            ycol = 1

        with file_path.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=self._delimiter)

            for idx, row in enumerate(csvReader):
                if row[0].startswith('#'):
                    continue

                if len(row) < 2:
                    raise ScanPointParseError('Bad number of columns!')

                point = ScanPoint(idx, float(row[xcol]), float(row[ycol]))
                pointList.append(point)

        return PositionSequence(pointList)


class DelimitedPositionFileWriter(PositionFileWriter):
    def __init__(self, delimiter: str, swapXY: bool) -> None:
        self._delimiter = delimiter
        self._swapXY = swapXY

    def write(self, file_path: Path, positions: PositionSequence) -> None:
        with file_path.open(mode='wt') as csvFile:
            for point in positions:
                x = point.position_x_m
                y = point.position_y_m
                line = (
                    f'{y}{self._delimiter}{x}\n' if self._swapXY else f'{x}{self._delimiter}{y}\n'
                )
                csvFile.write(line)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        DelimitedPositionFileReader(' ', swapXY=False),
        simple_name='TXT',
        display_name='Space-Separated Values Files (*.txt)',
    )
    registry.position_file_writers.register_plugin(
        DelimitedPositionFileWriter(' ', swapXY=False),
        simple_name='TXT',
        display_name='Space-Separated Values Files (*.txt)',
    )
    registry.position_file_readers.register_plugin(
        DelimitedPositionFileReader(',', swapXY=True),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
    registry.position_file_writers.register_plugin(
        DelimitedPositionFileWriter(',', swapXY=True),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
