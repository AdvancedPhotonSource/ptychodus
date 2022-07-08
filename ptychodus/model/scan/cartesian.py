from decimal import Decimal

from ...api.scan import ScanPoint
from .initializer import ScanInitializer, ScanInitializerParameters
from .settings import ScanSettings


class CartesianScanInitializer(ScanInitializer):

    def __init__(self, parameters: ScanInitializerParameters, stepSizeXInMeters: Decimal,
                 stepSizeYInMeters: Decimal, extentX: int, extentY: int, snake: bool) -> None:
        super().__init__(parameters)
        self._stepSizeXInMeters = stepSizeXInMeters
        self._stepSizeYInMeters = stepSizeYInMeters
        self._extentX = extentX
        self._extentY = extentY
        self._snake = snake

    @classmethod
    def createFromSettings(cls, parameters: ScanInitializerParameters, settings: ScanSettings,
                           snake: bool) -> CartesianScanInitializer:
        stepSizeXInMeters = settings.stepSizeXInMeters.value
        stepSizeYInMeters = settings.stepSizeYInMeters.value
        extentX = settings.extentX.value
        extentY = settings.extentY.value
        return cls(parameters, stepSizeXInMeters, stepSizeYInMeters, extentX, extentY, snake)

    def syncToSettings(self, settings: ScanSettings) -> None:
        settings.initializer.value = self.name
        settings.stepSizeXInMeters.value = self._stepSizeXInMeters
        settings.stepSizeYInMeters.value = self._stepSizeYInMeters
        settings.extentX.value = self._extentX
        settings.extentY.value = self._extentY
        super().syncToSettings(settings)

    @classmethod
    @property
    def category(self) -> str:
        return 'Cartesian'

    @classmethod
    @property
    def name(self) -> str:
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

    def getExtentX(self) -> int:
        return self._extentX

    def setExtentX(self, extentX: int) -> None:
        if self._extentX != extentX:
            self._extentX = extentX
            self.notifyObservers()

    def getExtentY(self) -> int:
        return self._extentY

    def setExtentY(self, extentY: int) -> None:
        if self._extentY != extentY:
            self._extentY = extentY
            self.notifyObservers()

    def _getPoint(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        y, x = divmod(index, self._nx)

        if self._snake and y & 1:
            x = self._extentX - 1 - x

        xf = x * self._stepSizeXInMeters
        yf = y * self._stepSizeYInMeters

        return ScanPoint(xf, yf)

    def __len__(self) -> int:
        return self._extentX * self._extentY
