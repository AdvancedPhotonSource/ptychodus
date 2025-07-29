from __future__ import annotations
from enum import IntEnum
from pathlib import Path
from typing import Final
import csv

import numpy

from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint, ScanPointParseError
from .aps33id_velociprobe_diffraction_file import VelociprobeDiffractionFileReader

__all__ = [
    'VelociprobePositionFileReader',
]


class VelociprobePositionFileColumn(IntEnum):
    X = 1
    LASER_INTERFEROMETER_Y = 2
    POSITION_ENCODER_Y = 5
    TRIGGER = 7


class VelociprobePositionFileReader(PositionFileReader):
    ONE_NANOMETER_M: Final[float] = 1.0e-9

    def __init__(self, diffraction_reader: VelociprobeDiffractionFileReader, y_column: int) -> None:
        self._diffraction_reader = diffraction_reader
        self._y_column = y_column

    @classmethod
    def create_laser_interferometer_instance(
        cls, diffraction_reader: VelociprobeDiffractionFileReader
    ) -> VelociprobePositionFileReader:
        return cls(diffraction_reader, VelociprobePositionFileColumn.LASER_INTERFEROMETER_Y)

    @classmethod
    def create_position_encoder_instance(
        cls, diffraction_reader: VelociprobeDiffractionFileReader
    ) -> VelociprobePositionFileReader:
        return cls(diffraction_reader, VelociprobePositionFileColumn.POSITION_ENCODER_Y)

    def _apply_transform(self, positions: PositionSequence) -> PositionSequence:
        stage_rotation_rad = numpy.deg2rad(self._diffraction_reader.stage_rotation_deg)
        stage_rotation_cos = numpy.cos(stage_rotation_rad)

        x_mean = sum(p.position_x_m for p in positions) / len(positions)
        y_mean = sum(p.position_y_m for p in positions) / len(positions)
        point_list: list[ScanPoint] = list()

        for untransformed_point in positions:
            point = ScanPoint(
                untransformed_point.index,
                (untransformed_point.position_x_m - x_mean) * stage_rotation_cos,
                (untransformed_point.position_y_m - y_mean),
            )
            point_list.append(point)

        return PositionSequence(point_list)

    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()
        minimum_column_count = max(col.value for col in VelociprobePositionFileColumn) + 1

        with file_path.open(newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                if row[0].startswith('#'):
                    continue

                if len(row) < minimum_column_count:
                    raise ScanPointParseError('Bad number of columns!')

                trigger = int(row[VelociprobePositionFileColumn.TRIGGER])
                x_nm = int(row[VelociprobePositionFileColumn.X])
                y_nm = int(row[self._y_column])

                if self._y_column == VelociprobePositionFileColumn.POSITION_ENCODER_Y:
                    y_nm = -y_nm

                point = ScanPoint(
                    trigger,
                    x_nm * self.ONE_NANOMETER_M,
                    y_nm * self.ONE_NANOMETER_M,
                )
                point_list.append(point)

        raw_positions = PositionSequence(point_list)

        return self._apply_transform(raw_positions)
