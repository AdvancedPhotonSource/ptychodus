from __future__ import annotations
from decimal import Decimal

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


class CartesianScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters, stepSizeXInMeters: Decimal,
                 stepSizeYInMeters: Decimal, numberOfPointsX: int, numberOfPointsY: int,
                 snake: bool) -> None:
        super().__init__(parameters)
        self._stepSizeXInMeters = stepSizeXInMeters
        self._stepSizeYInMeters = stepSizeYInMeters
        self._numberOfPointsX = numberOfPointsX
        self._numberOfPointsY = numberOfPointsY
        self._snake = snake

    @classmethod
    def createFromSettings(cls, parameters: ScanInitializerParameters, settings: ScanSettings,
                           snake: bool) -> CartesianScanInitializer:
        stepSizeXInMeters = settings.stepSizeXInMeters.value
        stepSizeYInMeters = settings.stepSizeYInMeters.value
        numberOfPointsX = settings.numberOfPointsX.value
        numberOfPointsY = settings.numberOfPointsY.value
        return cls(parameters, stepSizeXInMeters, stepSizeYInMeters, numberOfPointsX,
                   numberOfPointsY, snake)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.stepSizeXInMeters.value = self._stepSizeXInMeters
        settings.stepSizeYInMeters.value = self._stepSizeYInMeters
        settings.numberOfPointsX.value = self._numberOfPointsX
        settings.numberOfPointsY.value = self._numberOfPointsY
        super().syncToSettings(settings)

    @property
    def nameHint(self) -> str:
        return self.variant

    @property
    def category(self) -> str:
        return 'Cartesian'

    @property
    def variant(self) -> str:
        return 'Snake' if self._snake else 'Raster'

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

    def _getPoint(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        y, x = divmod(index, self._numberOfPointsX)

        if self._snake and y & 1:
            x = self._numberOfPointsX - 1 - x

        xf = x * self._stepSizeXInMeters
        yf = y * self._stepSizeYInMeters

        return ScanPoint(xf, yf)

    def __len__(self) -> int:
        return self._numberOfPointsX * self._numberOfPointsY
