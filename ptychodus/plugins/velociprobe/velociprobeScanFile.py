from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum
from pathlib import Path
from statistics import median
from typing import Final
import csv
import math

from .velociprobeDataFile import VelociprobeDataFileReader
from ptychodus.api.scan import (ScanDictionary, ScanFileReader, ScanFileWriter, ScanPoint,
                                ScanPointParseError, ScanPointSequence, SimpleScanDictionary)


class VelociprobeScanFileColumn(IntEnum):
    X = 1
    LASER_INTERFEROMETER_Y = 2
    ENCODER_Y = 5
    TRIGGER = 7


@dataclass(frozen=True)
class VelociprobeScanPoint:
    x_nm: int
    y_li_nm: int
    y_en_nm: int


class VelociprobeScanFileReader(ScanFileReader):
    EXPECTED_NUMBER_OF_COLUMNS: Final[int] = 8

    def __init__(self, dataFileReader: VelociprobeDataFileReader) -> None:
        self._dataFileReader = dataFileReader

    @property
    def simpleName(self) -> str:
        return 'Velociprobe'

    @property
    def fileFilter(self) -> str:
        return 'Velociprobe Scan Files (*.txt)'

    def _applyTransform(self, pointList: list[ScanPoint]) -> None:
        zero = Decimal()
        numberOfPoints = Decimal(len(pointList))

        xMean = sum([point.x for point in pointList], start=zero) / numberOfPoints
        yMean = sum([point.y for point in pointList], start=zero) / numberOfPoints

        stageRotationInRadians = self._dataFileReader.entry.sample.goniometer.chi_rad \
                if self._dataFileReader.entry else 0.
        stageRotationCosine = Decimal(math.cos(stageRotationInRadians))

        for idx, point in enumerate(pointList):
            x = (point.x - xMean) * stageRotationCosine
            y = (point.y - yMean)
            pointList[idx] = ScanPoint(x, y)

    def read(self, filePath: Path) -> ScanDictionary:
        pointDict: dict[int, list[VelociprobeScanPoint]] = defaultdict(list[VelociprobeScanPoint])
        liPointList: list[ScanPoint] = list()
        enPointList: list[ScanPoint] = list()

        with open(filePath, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter=',')

            for row in csvReader:
                if row[0].startswith('#'):
                    continue

                if len(row) != VelociprobeScanFileReader.EXPECTED_NUMBER_OF_COLUMNS:
                    raise ScanPointParseError('Bad number of columns!')

                x_nm = int(row[VelociprobeScanFileColumn.X])
                y_li_nm = int(row[VelociprobeScanFileColumn.LASER_INTERFEROMETER_Y])
                y_en_nm = int(row[VelociprobeScanFileColumn.ENCODER_Y])
                trigger = int(row[VelociprobeScanFileColumn.TRIGGER])

                point = VelociprobeScanPoint(x_nm, y_li_nm, -y_en_nm)
                pointDict[trigger].append(point)

        for trigger, pointList in pointDict.items():
            xf_nm = median([p.x_nm for p in pointList])
            yf_li_nm = median([p.y_li_nm for p in pointList])
            yf_en_nm = median([p.y_en_nm for p in pointList])

            nm_to_m = Decimal('1e-9')
            x_m = Decimal(xf_nm) * nm_to_m
            y_li_m = Decimal(yf_li_nm) * nm_to_m
            y_en_m = Decimal(yf_en_nm) * nm_to_m

            liPointList.append(ScanPoint(x_m, y_li_m))
            enPointList.append(ScanPoint(x_m, y_en_m))

        self._applyTransform(liPointList)
        self._applyTransform(enPointList)

        scanDict: dict[str, ScanPointSequence] = dict()
        scanDict['LaserInterferometerY'] = liPointList
        scanDict['EncoderY'] = enPointList
        return SimpleScanDictionary(scanDict)
