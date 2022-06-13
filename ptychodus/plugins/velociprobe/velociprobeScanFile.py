from collections import defaultdict
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from typing import Iterable
import csv

import numpy

from .velociprobeDataFile import VelociprobeDataFileReader
from ptychodus.api.scan import ScanFileReader, ScanPoint, ScanPointParseError


class VelociprobeScanYPositionSource(IntEnum):
    LASER_INTERFEROMETER = 2
    ENCODER = 5


class VelociprobeScanPointList:

    def __init__(self) -> None:
        self.xInNanometers: list[int] = list()
        self.yInNanometers: list[int] = list()

    def append(self, x_nm: int, y_nm: int) -> None:
        self.xInNanometers.append(x_nm)
        self.yInNanometers.append(y_nm)

    def mean(self) -> ScanPoint:
        nanometersToMeters = Decimal('1e-9')
        x_nm = Decimal(sum(self.xInNanometers)) / Decimal(len(self.xInNanometers))
        y_nm = Decimal(sum(self.yInNanometers)) / Decimal(len(self.yInNanometers))
        return ScanPoint(x_nm * nanometersToMeters, y_nm * nanometersToMeters)


class VelociprobeScanFileReader(ScanFileReader):
    X_COLUMN = 1
    TRIGGER_COLUMN = 7

    def __init__(self, dataFileReader: VelociprobeDataFileReader,
                 yPositionSource: VelociprobeScanYPositionSource) -> None:
        self._dataFileReader = dataFileReader
        self._yPositionSource = yPositionSource

    @property
    def simpleName(self) -> str:
        yPositionSourceText = 'EncoderY'

        if self._yPositionSource == VelociprobeScanYPositionSource.LASER_INTERFEROMETER:
            yPositionSourceText = 'LaserInterferometerY'

        return f'VelociprobeWith{yPositionSourceText}'

    @property
    def fileFilter(self) -> str:
        yPositionSourceText = 'Encoder Y'

        if self._yPositionSource == VelociprobeScanYPositionSource.LASER_INTERFEROMETER:
            yPositionSourceText = 'Laser Interferometer Y'

        return f'Velociprobe Scan Files - {yPositionSourceText} (*.txt)'

    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        pointDict: dict[int, VelociprobeScanPointList] = defaultdict(VelociprobeScanPointList)

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) != 8:
                    raise ScanPointParseError()

                trigger = int(row[VelociprobeScanFileReader.TRIGGER_COLUMN])
                x_nm = int(row[VelociprobeScanFileReader.X_COLUMN])
                y_nm = int(row[self._yPositionSource.value])

                if self._yPositionSource == VelociprobeScanYPositionSource.ENCODER:
                    y_nm = -y_nm

                pointDict[trigger].append(x_nm, y_nm)

        pointList = [pointList.mean() for _, pointList in sorted(pointDict.items())]
        xMeanInMeters = Decimal(sum(point.x for point in pointList)) / len(pointList)
        yMeanInMeters = Decimal(sum(point.y for point in pointList)) / len(pointList)

        for idx, point in enumerate(pointList):
            stageRotationInRadians = 0.

            if self._dataFileReader.entry:
                stageRotationInRadians = self._dataFileReader.entry.sample.goniometer.chi_rad

            xInMeters = (point.x - xMeanInMeters) * Decimal(numpy.cos(stageRotationInRadians))
            yInMeters = (point.y - yMeanInMeters)
            pointList[idx] = ScanPoint(xInMeters, yInMeters)

        return pointList
