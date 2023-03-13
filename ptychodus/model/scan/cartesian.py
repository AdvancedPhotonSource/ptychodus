from collections.abc import Iterator
from decimal import Decimal
from typing import Final

from ...api.scan import ScanPoint
from .itemRepository import ScanRepositoryItem
from .settings import ScanSettings


class CartesianScanRepositoryItem(ScanRepositoryItem):

    def __init__(self, snake: bool) -> None:
        super().__init__()
        self._stepSizeXInMeters = Decimal()
        self._stepSizeYInMeters = Decimal()
        self._numberOfPointsX = 0
        self._numberOfPointsY = 0
        self._snake = snake

    @property
    def name(self) -> str:
        return self.variant

    @property
    def category(self) -> str:
        return 'Cartesian'

    @property
    def canActivate(self) -> bool:
        return True

    def syncFromSettings(self, settings: ScanSettings) -> None:
        self._stepSizeXInMeters = settings.stepSizeXInMeters.value
        self._stepSizeYInMeters = settings.stepSizeYInMeters.value
        self._numberOfPointsX = settings.numberOfPointsX.value
        self._numberOfPointsY = settings.numberOfPointsY.value
        self.notifyObservers()

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.stepSizeXInMeters.value = self._stepSizeXInMeters
        settings.stepSizeYInMeters.value = self._stepSizeYInMeters
        settings.numberOfPointsX.value = self._numberOfPointsX
        settings.numberOfPointsY.value = self._numberOfPointsY

    def __iter__(self) -> Iterator[int]:
        for index in range(len(self)):
            yield index

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        y, x = divmod(index, self._numberOfPointsX)

        if self._snake and y & 1:
            x = self._numberOfPointsX - 1 - x

        center = self._getIndexCenter()

        xf = (x - center.x) * self._stepSizeXInMeters
        yf = (y - center.y) * self._stepSizeYInMeters

        return ScanPoint(xf, yf)

    def __len__(self) -> int:
        return self._numberOfPointsX * self._numberOfPointsY

    def getStepSizeXInMeters(self) -> Decimal:
        return self._stepSizeXInMeters

    def setStepSizeXInMeters(self, stepSizeXInMeters: Decimal) -> None:
        if self._stepSizeXInMeters != stepSizeXInMeters:
            self._stepSizeXInMeters = stepSizeXInMeters
            self.notifyObservers()

    def getStepSizeYInMeters(self) -> Decimal:
        return self._stepSizeYInMeters

    def setStepSizeYInMeters(self, stepSizeYInMeters: Decimal) -> None:
        if self._stepSizeYInMeters != stepSizeYInMeters:
            self._stepSizeYInMeters = stepSizeYInMeters
            self.notifyObservers()

    def getNumberOfPointsX(self) -> int:
        return self._numberOfPointsX

    def setNumberOfPointsX(self, numberOfPointsX: int) -> None:
        if self._numberOfPointsX != numberOfPointsX:
            self._numberOfPointsX = numberOfPointsX
            self.notifyObservers()

    def getNumberOfPointsY(self) -> int:
        return self._numberOfPointsY

    def setNumberOfPointsY(self, numberOfPointsY: int) -> None:
        if self._numberOfPointsY != numberOfPointsY:
            self._numberOfPointsY = numberOfPointsY
            self.notifyObservers()

    def _getIndexCenter(self) -> ScanPoint:
        return ScanPoint(
            Decimal(self._numberOfPointsX - 1) / 2,
            Decimal(self._numberOfPointsY - 1) / 2,
        )


class RasterScanRepositoryItem(CartesianScanRepositoryItem):
    NAME: Final[str] = 'Raster'

    def __init__(self) -> None:
        super().__init__(snake=False)

    @property
    def variant(self) -> str:
        return self.NAME


class SnakeScanRepositoryItem(CartesianScanRepositoryItem):
    NAME: Final[str] = 'Snake'

    def __init__(self) -> None:
        super().__init__(snake=True)

    @property
    def variant(self) -> str:
        return self.NAME
