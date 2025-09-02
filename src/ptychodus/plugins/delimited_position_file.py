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
    def __init__(self, delimiter: str, *, swap_xy: bool) -> None:
        self._delimiter = delimiter
        self._swap_xy = swap_xy

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        if self._swap_xy:
            xcol = 1
            ycol = 0
        else:
            xcol = 0
            ycol = 1

        with file_path.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=self._delimiter)

            for idx, row in enumerate(csv_reader):
                if row[0].startswith('#'):
                    continue

                if len(row) < 2:
                    raise ScanPointParseError('Bad number of columns!')

                point = ScanPoint(idx, float(row[xcol]), float(row[ycol]))
                point_list.append(point)

        return PositionSequence(point_list)


class DelimitedPositionFileWriter(PositionFileWriter):
    def __init__(self, delimiter: str, swap_xy: bool) -> None:
        self._delimiter = delimiter
        self._swap_xy = swap_xy

    def write(self, file_path: Path, positions: PositionSequence) -> None:
        with file_path.open(mode='wt') as csv_file:
            for point in positions:
                x = point.position_x_m
                y = point.position_y_m
                line = (
                    f'{y}{self._delimiter}{x}\n' if self._swap_xy else f'{x}{self._delimiter}{y}\n'
                )
                csv_file.write(line)


def register_plugins(registry: PluginRegistry) -> None:
    registry.position_file_readers.register_plugin(
        DelimitedPositionFileReader(' ', swap_xy=False),
        simple_name='TXT',
        display_name='Space-Separated Values Files (*.txt)',
    )
    registry.position_file_writers.register_plugin(
        DelimitedPositionFileWriter(' ', swap_xy=False),
        simple_name='TXT',
        display_name='Space-Separated Values Files (*.txt)',
    )
    registry.position_file_readers.register_plugin(
        DelimitedPositionFileReader(',', swap_xy=True),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
    registry.position_file_writers.register_plugin(
        DelimitedPositionFileWriter(',', swap_xy=True),
        simple_name='CSV',
        display_name='Comma-Separated Values Files (*.csv)',
    )
