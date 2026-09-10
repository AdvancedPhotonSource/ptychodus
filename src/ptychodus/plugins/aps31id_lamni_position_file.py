"""Probe position reader for APS 31-ID-E LamNI ``*.dat`` scan-position files.

LamNI writes positions as space-delimited text with a title line, a column-header
line, then one row per sample. Two data-acquisition paths produce three header
layouts, and this reader selects among them by matching the header row exactly.
The split is by DAQ path, not by scan mode: the Orchestra layout covers both step
and fly scans.

Index origin
------------
Positions are joined to diffraction patterns by integer index, and the LamNI
diffraction reader indexes patterns positionally from zero
(``numpy.arange(num_patterns)``). The two DAQ paths do not agree on where their
own counters start:

- Orchestra's ``DataPoint`` is the row position, running ``0..N-1``, so it lines
  up with the pattern indexes as written.
- softGlueZynq's ``Detector_Count`` is a detector frame counter running ``1..N``.
  In the raw layout it is not a row position at all -- several oversampled rows
  share one value, and the duplicates are averaged into a single anchor
  downstream by ``prepare_reconstruct_input``.

On one scan recorded simultaneously through both paths, ``Detector_Count`` equals
``DataPoint + 1`` on every row. Each format therefore declares the counter value
of the first detector frame in ``index_origin``, which is subtracted to yield a
zero-based index. The subtraction for softGlueZynq is deliberate: without it,
every position pairs with the following frame and the first pattern is dropped
for falling outside the position range.
"""

from __future__ import annotations
from dataclasses import dataclass
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


@dataclass(frozen=True)
class _PositionFormat:
    """One known LamNI column layout and where its index and coordinates live."""

    name: str
    header: tuple[str, ...]
    index_column: int
    index_origin: int  # index-column value of the first detector frame
    x_column: int
    y_column: int


_ORCHESTRA: Final = _PositionFormat(
    name='Orchestra',
    header=(
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
    ),
    index_column=0,
    index_origin=0,
    x_column=3,
    y_column=6,
)

_SOFT_GLUE_ZYNQ_RAW: Final = _PositionFormat(
    name='softGlueZynq raw',
    header=(
        'DataPoint',
        'x_st_fzp',
        'y_st_fzp',
        'ckUser_Clk_Count',
        'Detector_Count',
    ),
    index_column=4,
    index_origin=1,
    x_column=1,
    y_column=2,
)

_SOFT_GLUE_ZYNQ_PROCESSED: Final = _PositionFormat(
    name='softGlueZynq processed',
    header=(
        'Detector_Count',
        'Average_x_st_fzp',
        'Stdev_x_st_fzp',
        'Average_y_st_fzp',
        'Stdev_y_st_fzp',
    ),
    index_column=0,
    index_origin=1,
    x_column=1,
    y_column=3,
)

_FORMATS: Final = (_ORCHESTRA, _SOFT_GLUE_ZYNQ_RAW, _SOFT_GLUE_ZYNQ_PROCESSED)


def _match_format(header_row: list[str]) -> _PositionFormat | None:
    header = tuple(header_row)

    for format_ in _FORMATS:
        if header == format_.header:
            return format_

    return None


def _describe_known_formats() -> str:
    return '\n'.join(f'  {format_.name}: {" ".join(format_.header)}' for format_ in _FORMATS)


class LamNIPositionFileReader(ProbePositionFileReader):
    """Reader for APS 31-ID-E LamNI scan-position files, sensing the layout by header."""

    SIMPLE_NAME: Final[str] = 'APS_LamNI'
    DISPLAY_NAME: Final[str] = 'APS 31-ID-E LamNI Position Files (*.dat)'

    def read(self, file_path: Path) -> ProbePositionSequence:
        point_list: list[ProbePosition] = list()

        with file_path.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ', skipinitialspace=True)
            csv_iterator = iter(csv_reader)

            title_row = next(csv_iterator)

            try:
                scan_name = ' '.join(title_row).split(',', maxsplit=1)[0]
            except IndexError:
                raise ProbePositionParseError('Bad scan name!')

            column_header_row = next(csv_iterator)
            format_ = _match_format(column_header_row)

            if format_ is None:
                raise ProbePositionParseError(
                    'Bad LamNI header!\n'
                    f'Found:    {" ".join(column_header_row)}\n'
                    f'Expected one of:\n{_describe_known_formats()}\n'
                )

            logger.debug(f'Reading {format_.name} scan positions for "{scan_name}"...')

            for row in csv_iterator:
                if row[0].startswith('#'):
                    continue

                if len(row) != len(format_.header):
                    raise ProbePositionParseError('Bad number of columns!')

                point = ProbePosition(
                    int(row[format_.index_column]) - format_.index_origin,
                    -LengthUnit.MICROMETER.to_meters(float(row[format_.x_column])),
                    -LengthUnit.MICROMETER.to_meters(float(row[format_.y_column])),
                )
                point_list.append(point)

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        LamNIPositionFileReader(),
        simple_name=LamNIPositionFileReader.SIMPLE_NAME,
        display_name=LamNIPositionFileReader.DISPLAY_NAME,
    )
