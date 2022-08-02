from decimal import Decimal

from ...api.geometry import Interval
from ...api.observer import Observable


class DisplayRange(Observable):

    @staticmethod
    def createUnitInterval() -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def __init__(self) -> None:
        super().__init__()
        self._range = DisplayRange.createUnitInterval()
        self._limits = DisplayRange.createUnitInterval()

    def getLimits(self) -> Interval[Decimal]:
        return self._limits

    def setLimits(self, lower: Decimal, upper: Decimal) -> None:
        self._limits = Interval[Decimal](lower, upper)
        self.notifyObservers()

    def getLower(self) -> Decimal:
        return self._limits.clamp(self._range.lower)

    def setLower(self, value: Decimal) -> None:
        self._range.lower = value
        self.notifyObservers()

    def getUpper(self) -> Decimal:
        return self._limits.clamp(self._range.upper)

    def setUpper(self, value: Decimal) -> None:
        self._range.upper = value
        self.notifyObservers()

    def setRangeAndLimits(self, interval: Interval[Decimal]) -> None:
        self._range = interval.copy()
        self._limits = interval.copy()
        self.notifyObservers()
