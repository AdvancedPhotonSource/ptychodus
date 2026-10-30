from pathlib import Path
from typing import Final
import csv

from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ProbePositionFileReader,
    ProbePositionFileWriter,
    ProbePosition,
    ProbePositionParseError,
)


class DelimitedPositionFileReader(ProbePositionFileReader):
    def __init__(self, delimiter: str, *, swap_xy: bool, description: str, extension: str) -> None:
        self._delimiter = delimiter
        self._swap_xy = swap_xy
        self._description = description
        self._extension = extension

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return {self._extension}

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

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
                    raise ProbePositionParseError('Bad number of columns!')

                point = ProbePosition(idx, float(row[xcol]), float(row[ycol]))
                point_list.append(point)

        return ProbePositionSequence(point_list)


class DelimitedPositionFileWriter(ProbePositionFileWriter):
    def __init__(self, delimiter: str, swap_xy: bool, *, description: str, extension: str) -> None:
        self._delimiter = delimiter
        self._swap_xy = swap_xy
        self._description = description
        self._extension = extension

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return {self._extension}

    def write(self, file_path: Path, positions: ProbePositionSequence) -> None:
        with file_path.open(mode='wt') as csv_file:
            for point in positions:
                x = point.x_m
                y = point.y_m
                line = (
                    f'{y}{self._delimiter}{x}\n' if self._swap_xy else f'{x}{self._delimiter}{y}\n'
                )
                csv_file.write(line)


_SPACE_DESCRIPTION: Final[str] = 'Space-Separated Values Files'
_COMMA_DESCRIPTION: Final[str] = 'Comma-Separated Values Files'


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        DelimitedPositionFileReader(
            ' ', swap_xy=False, description=_SPACE_DESCRIPTION, extension='.txt'
        ),
        simple_name='TXT',
    )
    registry.probe_position_file_writers.register_plugin(
        DelimitedPositionFileWriter(' ', False, description=_SPACE_DESCRIPTION, extension='.txt'),
        simple_name='TXT',
    )
    registry.probe_position_file_readers.register_plugin(
        DelimitedPositionFileReader(
            ',', swap_xy=True, description=_COMMA_DESCRIPTION, extension='.csv'
        ),
        simple_name='CSV',
    )
    registry.probe_position_file_writers.register_plugin(
        DelimitedPositionFileWriter(',', True, description=_COMMA_DESCRIPTION, extension='.csv'),
        simple_name='CSV',
    )
