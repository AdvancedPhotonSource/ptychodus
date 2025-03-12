from __future__ import annotations
from enum import IntEnum
from pathlib import Path
from typing import Final
import csv

import numpy

from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError
from .neXusDiffractionFile import NeXusDiffractionFileReader

__all__ = [
    'VelociprobeScanFileReader',
]


class VelociprobeScanFileColumn(IntEnum):
    X = 1
    LASER_INTERFEROMETER_Y = 2
    POSITION_ENCODER_Y = 5
    TRIGGER = 7


class VelociprobeScanFileReader(ScanFileReader):
    NANOMETERS_TO_METERS: Final[float] = 1.0e-9

    def __init__(self, neXusReader: NeXusDiffractionFileReader, yColumn: int) -> None:
        self._neXusReader = neXusReader
        self._yColumn = yColumn

    @classmethod
    def createLaserInterferometerInstance(
        cls, neXusReader: NeXusDiffractionFileReader
    ) -> VelociprobeScanFileReader:
        return cls(neXusReader, VelociprobeScanFileColumn.LASER_INTERFEROMETER_Y)

    @classmethod
    def createPositionEncoderInstance(
        cls, neXusReader: NeXusDiffractionFileReader
    ) -> VelociprobeScanFileReader:
        return cls(neXusReader, VelociprobeScanFileColumn.POSITION_ENCODER_Y)

    def _applyTransform(self, scan: Scan) -> Scan:
        stageRotationInRadians = numpy.deg2rad(self._neXusReader.stageRotationInDegrees)
        stageRotationCosine = numpy.cos(stageRotationInRadians)

        xMean = sum(p.position_x_m for p in scan) / len(scan)
        yMean = sum(p.position_y_m for p in scan) / len(scan)
        pointList: list[ScanPoint] = list()

        for untransformedPoint in scan:
            point = ScanPoint(
                untransformedPoint.index,
                (untransformedPoint.position_x_m - xMean) * stageRotationCosine,
                (untransformedPoint.position_y_m - yMean),
            )
            pointList.append(point)

        return Scan(pointList)

    def read(self, filePath: Path) -> Scan:
        pointList: list[ScanPoint] = list()
        minimumColumnCount = max(col.value for col in VelociprobeScanFileColumn) + 1

        with filePath.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < minimumColumnCount:
                    raise ScanPointParseError('Bad number of columns!')

                trigger = int(row[VelociprobeScanFileColumn.TRIGGER])
                x_nm = int(row[VelociprobeScanFileColumn.X])
                y_nm = int(row[self._yColumn])

                if self._yColumn == VelociprobeScanFileColumn.POSITION_ENCODER_Y:
                    y_nm = -y_nm

                point = ScanPoint(
                    trigger,
                    x_nm * self.NANOMETERS_TO_METERS,
                    y_nm * self.NANOMETERS_TO_METERS,
                )
                pointList.append(point)

        rawScan = Scan(pointList)

        return self._applyTransform(rawScan)
