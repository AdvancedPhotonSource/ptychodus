from collections.abc import Iterator
from decimal import Decimal
from typing import Final

import numpy

from ...api.scan import Scan, ScanPoint
from .repository import ScanInitializer
from .settings import ScanSettings

__all__ = [
    'SpiralScanInitializer',
]


class SpiralScan(Scan):
    NAME: Final[str] = 'Spiral'
    '''https://doi.org/10.1364/OE.22.012634'''

    def __init__(self) -> None:
        self.numberOfPoints = 100
        self.radiusScalarInMeters = Decimal('5e-7')

    @property
    def nameHint(self) -> str:
        return 'Fermat'

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield index

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        radiusInMeters = self.radiusScalarInMeters * Decimal(index).sqrt()
        divergenceAngleInRadians = (3. - numpy.sqrt(5)) * numpy.pi
        thetaInRadians = divergenceAngleInRadians * index

        return ScanPoint(
            radiusInMeters * Decimal(numpy.cos(thetaInRadians)),
            radiusInMeters * Decimal(numpy.sin(thetaInRadians)),
        )

    def __len__(self) -> int:
        return self.numberOfPoints


class SpiralScanInitializer(ScanInitializer):
    SIMPLE_NAME: Final[str] = SpiralScan.NAME
    DISPLAY_NAME: Final[str] = SpiralScan.NAME

    def __init__(self) -> None:
        super().__init__()
        self._scan = SpiralScan()

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._scan.numberOfPoints = settings.numberOfPointsX.value * settings.numberOfPointsY.value
        self._scan.radiusScalarInMeters = settings.radiusScalarInMeters.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.numberOfPointsX.value = self._scan.numberOfPoints
        settings.numberOfPointsY.value = 1
        settings.radiusScalarInMeters.value = self._scan.radiusScalarInMeters

    def __call__(self) -> Scan:
        return self._scan

    def getNumberOfPoints(self) -> int:
        return self._scan.numberOfPoints

    def setNumberOfPoints(self, numberOfPoints: int) -> None:
        if self._scan.numberOfPoints != numberOfPoints:
            self._scan.numberOfPoints = numberOfPoints
            self.notifyObservers()

    def getRadiusScalarInMeters(self) -> Decimal:
        return self._scan.radiusScalarInMeters

    def setRadiusScalarInMeters(self, radiusScalarInMeters: Decimal) -> None:
        if self._scan.radiusScalarInMeters != radiusScalarInMeters:
            self._scan.radiusScalarInMeters = radiusScalarInMeters
            self.notifyObservers()
