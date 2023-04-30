from __future__ import annotations
from collections import defaultdict
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
import csv

import numpy

from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan
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

    def __init__(self, neXusReader: NeXusDiffractionFileReader, yColumn: int) -> None:
        self._neXusReader = neXusReader
        self._yColumn = yColumn

    @classmethod
    def createLaserInterferometerInstance(
            cls, neXusReader: NeXusDiffractionFileReader) -> VelociprobeScanFileReader:
        return cls(neXusReader, VelociprobeScanFileColumn.LASER_INTERFEROMETER_Y)

    @classmethod
    def createPositionEncoderInstance(
            cls, neXusReader: NeXusDiffractionFileReader) -> VelociprobeScanFileReader:
        return cls(neXusReader, VelociprobeScanFileColumn.POSITION_ENCODER_Y)

    @property
    def simpleName(self) -> str:
        name = 'VelociprobeUnknown'

        if self._yColumn == VelociprobeScanFileColumn.LASER_INTERFEROMETER_Y:
            name = 'VelociprobeLaserInterferometer'
        elif self._yColumn == VelociprobeScanFileColumn.POSITION_ENCODER_Y:
            name = 'VelociprobePositionEncoder'

        return name

    @property
    def fileFilter(self) -> str:
        ySource = 'Unknown'

        if self._yColumn == VelociprobeScanFileColumn.LASER_INTERFEROMETER_Y:
            ySource = 'Laser Interferometer'
        elif self._yColumn == VelociprobeScanFileColumn.POSITION_ENCODER_Y:
            ySource = 'Position Encoder'

        return f'Velociprobe Scan Files - {ySource} (*.txt)'

    def _applyTransform(self, scan: Scan) -> Scan:
        stageRotationInRadians = numpy.deg2rad(self._neXusReader.stageRotationInDegrees)
        stageRotationCosine = Decimal(repr(numpy.cos(stageRotationInRadians)))

        zero = Decimal()
        xMean = sum((p.x for p in scan.values()), start=zero) / len(scan)
        yMean = sum((p.y for p in scan.values()), start=zero) / len(scan)
        pointMap: dict[int, ScanPoint] = dict()

        for index, point in scan.items():
            pointMap[index] = ScanPoint(
                x=(point.x - xMean) * stageRotationCosine,
                y=(point.y - yMean),
            )

        return TabularScan(pointMap)

    def read(self, filePath: Path) -> Scan:
        pointSeqMap: dict[int, list[ScanPoint]] = defaultdict(list[ScanPoint])
        minimumColumnCount = max(col.value for col in VelociprobeScanFileColumn) + 1
        nanometersToMeters = Decimal('1e-9')

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
                    x=x_nm * nanometersToMeters,
                    y=y_nm * nanometersToMeters,
                )
                pointSeqMap[trigger].append(point)

        rawScan = TabularScan.createFromMappedPointIterable(pointSeqMap)

        return self._applyTransform(rawScan)
