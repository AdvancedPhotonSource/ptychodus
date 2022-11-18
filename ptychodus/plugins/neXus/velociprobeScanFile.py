from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from statistics import median
import csv

import numpy

from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint, ScanPointParseError, TabularScan
from ptychodus.api.plugins import PluginRegistry


class VelociprobeScanFileColumn(IntEnum):
    X = 1
    LASER_INTERFEROMETER_Y = 2
    ENCODER_Y = 5
    TRIGGER = 7


class VelociprobeScanFileReader(ScanFileReader):

    def __init__(self) -> None:
        self._stageRotationCosine = Decimal(1)

    @property
    def simpleName(self) -> str:
        return 'Velociprobe'

    @property
    def fileFilter(self) -> str:
        return 'Velociprobe Scan Files (*.txt)'

    def setStageRotationInDegrees(self, degrees: float) -> None:
        radians = numpy.deg2rad(degrees)
        cosine = numpy.cos(radians)
        self._stageRotationCosine = Decimal(repr(cosine))

    def _applyTransform(self, pointDict: dict[int, list[ScanPoint]]) -> None:
        xValues = [median(p.x for p in pl) for pl in pointDict.values()]
        yValues = [median(p.y for p in pl) for pl in pointDict.values()]

        zero = Decimal()
        xMean = sum(xValues, start=zero) / len(xValues)
        yMean = sum(yValues, start=zero) / len(yValues)

        for pointList in pointDict.values():
            for idx, point in enumerate(pointList):
                pointList[idx] = ScanPoint(
                    x=(point.x - xMean) * self._stageRotationCosine,
                    y=(point.y - yMean),
                )

    def read(self, filePath: Path) -> Sequence[Scan]:
        enPointDict: dict[int, list[ScanPoint]] = defaultdict(list[ScanPoint])
        liPointDict: dict[int, list[ScanPoint]] = defaultdict(list[ScanPoint])
        minimumColumnCount = max(col.value for col in VelociprobeScanFileColumn) + 1
        nanometersToMeters = Decimal('1e-9')

        with filePath.open(newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) < minimumColumnCount:
                    raise ScanPointParseError('Bad number of columns!')

                x_nm = int(row[VelociprobeScanFileColumn.X])
                y_li_nm = int(row[VelociprobeScanFileColumn.LASER_INTERFEROMETER_Y])
                y_en_nm = int(row[VelociprobeScanFileColumn.ENCODER_Y])
                trigger = int(row[VelociprobeScanFileColumn.TRIGGER])

                liPoint = ScanPoint(
                    x=x_nm * nanometersToMeters,
                    y=+y_li_nm * nanometersToMeters,
                )
                liPointDict[trigger].append(liPoint)

                enPoint = ScanPoint(
                    x=x_nm * nanometersToMeters,
                    y=-y_en_nm * nanometersToMeters,
                )
                enPointDict[trigger].append(enPoint)

        self._applyTransform(liPointDict)
        self._applyTransform(enPointDict)

        return [
            TabularScan('VelociprobeLaserInterferometerY', liPointDict),
            TabularScan('VelociprobeEncoderY', enPointDict),
        ]
