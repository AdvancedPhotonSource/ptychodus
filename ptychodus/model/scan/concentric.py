from collections.abc import Iterator
from decimal import Decimal
from typing import Final

import numpy

#from ...api.scan import Scan, ScanPoint
#from .repository import ScanInitializer
#from .settings import ScanSettings
from ptychodus.api.scan import Scan, ScanPoint
from ptychodus.model.scan.repository import ScanInitializer
from ptychodus.model.scan.settings import ScanSettings

__all__ = [
    'ConcentricScanInitializer',
]


class ConcentricScan(Scan):
    NAME: Final[str] = 'Concentric'
    '''https://doi.org/10.1088/1367-2630/12/3/035017'''

    def __init__(self) -> None:
        self.radialStepSizeInMeters = Decimal('1e-6')
        self.numberOfShells = 5
        self.numberOfPointsInFirstShell = 10

    @property
    def nameHint(self) -> str:
        return 'Concentric'

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield index

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        triangle = index // self.numberOfPointsInFirstShell
        shellIndex = int((1 + numpy.sqrt(1 + 8 * triangle)) / 2) - 1  # see OEIS A002024
        shellTriangle = (shellIndex * (shellIndex + 1)) // 2
        firstIndexInShell = self.numberOfPointsInFirstShell * shellTriangle
        pointIndexInShell = index - firstIndexInShell

        radiusInMeters = self.radialStepSizeInMeters * (shellIndex + 1)
        numberOfPointsInShell = self.numberOfPointsInFirstShell * (shellIndex + 1)
        thetaInRadians = 2 * numpy.pi * pointIndexInShell / numberOfPointsInShell

        return ScanPoint(
            radiusInMeters * Decimal(numpy.cos(thetaInRadians)),
            radiusInMeters * Decimal(numpy.sin(thetaInRadians)),
        )

    def __len__(self) -> int:
        triangle = (self.numberOfShells * (self.numberOfShells + 1)) // 2
        return triangle * self.numberOfPointsInFirstShell


class ConcentricScanInitializer(ScanInitializer):
    SIMPLE_NAME: Final[str] = ConcentricScan.NAME
    DISPLAY_NAME: Final[str] = ConcentricScan.NAME

    def __init__(self) -> None:
        super().__init__()
        self._scan = ConcentricScan()

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._scan.radialStepSizeInMeters = settings.radialStepSizeInMeters.value
        self._scan.numberOfShells = settings.numberOfShells.value
        self._scan.numberOfPointsInFirstShell = settings.numberOfPointsInFirstShell.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.radialStepSizeInMeters.value = self._scan.radialStepSizeInMeters
        settings.numberOfShells.value = self._scan.numberOfShells
        settings.numberOfPointsInFirstShell.value = self._scan.numberOfPointsInFirstShell

    def __call__(self) -> Scan:
        return self._scan

    def getRadialStepSizeInMeters(self) -> Decimal:
        return self._scan.radialStepSizeInMeters

    def setRadialStepSizeInMeters(self, radialStepSizeInMeters: Decimal) -> None:
        if self._scan.radialStepSizeInMeters != radialStepSizeInMeters:
            self._scan.radialStepSizeInMeters = radialStepSizeInMeters
            self.notifyObservers()

    def getNumberOfShells(self) -> int:
        return self._scan.numberOfShells

    def setNumberOfShells(self, numberOfShells: int) -> None:
        if self._scan.numberOfShells != numberOfShells:
            self._scan.numberOfShells = numberOfShells
            self.notifyObservers()

    def getNumberOfPointsInFirstShell(self) -> int:
        return self._scan.numberOfPointsInFirstShell

    def setNumberOfPointsInFirstShell(self, numberOfPointsInFirstShell: int) -> None:
        if self._scan.numberOfPointsInFirstShell != numberOfPointsInFirstShell:
            self._scan.numberOfPointsInFirstShell = numberOfPointsInFirstShell
            self.notifyObservers()
