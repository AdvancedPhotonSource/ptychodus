from collections.abc import Iterator
from decimal import Decimal

from ...api.scan import Scan, ScanPoint
from .repository import ScanInitializer
from .settings import ScanSettings

__all__ = [
    'CartesianScanInitializer',
]


class CartesianScan(Scan):

    def __init__(self, *, snake: bool, centered: bool) -> None:
        self.stepSizeXInMeters = Decimal('1e-6')
        self.stepSizeYInMeters = Decimal('1e-6')
        self.numberOfPointsX = 10
        self.numberOfPointsY = 10
        self._snake = snake
        self._centered = centered
        self._nameHint = ' '.join((
            'Centered' if centered else '',
            'Snake' if snake else 'Raster',
        ))

    @property
    def nameHint(self) -> str:
        return self._nameHint

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield index

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        y, x = divmod(index, self.numberOfPointsX)

        if self._snake:
            if y & 1:
                x = self.numberOfPointsX - 1 - x

        cx = (self.numberOfPointsX - 1) / 2
        cy = (self.numberOfPointsY - 1) / 2

        xf = (x - cx) * float(self.stepSizeXInMeters)
        yf = (y - cy) * float(self.stepSizeYInMeters)

        if self._centered:
            if y & 1:
                xf += float(self.stepSizeXInMeters) / 4
            else:
                xf -= float(self.stepSizeXInMeters) / 4

        return ScanPoint(xf, yf)

    def __len__(self) -> int:
        return self.numberOfPointsX * self.numberOfPointsY


class CartesianScanInitializer(ScanInitializer):

    def __init__(self, *, snake: bool, centered: bool) -> None:
        super().__init__()
        self._scan = CartesianScan(snake=snake, centered=centered)

    @property
    def simpleName(self) -> str:
        return ''.join(self.displayName.split())

    @property
    def displayName(self) -> str:
        return self._scan.nameHint

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._scan.stepSizeXInMeters = settings.stepSizeXInMeters.value
        self._scan.stepSizeYInMeters = settings.stepSizeYInMeters.value
        self._scan.numberOfPointsX = settings.numberOfPointsX.value
        self._scan.numberOfPointsY = settings.numberOfPointsY.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.stepSizeXInMeters.value = self._scan.stepSizeXInMeters
        settings.stepSizeYInMeters.value = self._scan.stepSizeYInMeters
        settings.numberOfPointsX.value = self._scan.numberOfPointsX
        settings.numberOfPointsY.value = self._scan.numberOfPointsY

    def __call__(self) -> Scan:
        return self._scan

    def getStepSizeXInMeters(self) -> Decimal:
        return self._scan.stepSizeXInMeters

    def setStepSizeXInMeters(self, stepSizeXInMeters: Decimal) -> None:
        if self._scan.stepSizeXInMeters != stepSizeXInMeters:
            self._scan.stepSizeXInMeters = stepSizeXInMeters
            self.notifyObservers()

    def getStepSizeYInMeters(self) -> Decimal:
        return self._scan.stepSizeYInMeters

    def setStepSizeYInMeters(self, stepSizeYInMeters: Decimal) -> None:
        if self._scan.stepSizeYInMeters != stepSizeYInMeters:
            self._scan.stepSizeYInMeters = stepSizeYInMeters
            self.notifyObservers()

    def getNumberOfPointsX(self) -> int:
        return self._scan.numberOfPointsX

    def setNumberOfPointsX(self, numberOfPointsX: int) -> None:
        if self._scan.numberOfPointsX != numberOfPointsX:
            self._scan.numberOfPointsX = numberOfPointsX
            self.notifyObservers()

    def getNumberOfPointsY(self) -> int:
        return self._scan.numberOfPointsY

    def setNumberOfPointsY(self, numberOfPointsY: int) -> None:
        if self._scan.numberOfPointsY != numberOfPointsY:
            self._scan.numberOfPointsY = numberOfPointsY
            self.notifyObservers()
