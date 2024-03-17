from ...api.geometry import Interval
from ...api.observer import Observable


class DisplayRange(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._range = Interval[float](0., 1.)
        self._limits = Interval[float](0., 1.)

    def getLimits(self) -> Interval[float]:
        return self._limits

    def setLimits(self, lower: float, upper: float) -> None:
        self._limits = Interval[float](lower, upper)
        self.notifyObservers()

    def getLower(self) -> float:
        return self._limits.clamp(self._range.lower)

    def setLower(self, value: float) -> None:
        self._range.lower = value
        self.notifyObservers()

    def getUpper(self) -> float:
        return self._limits.clamp(self._range.upper)

    def setUpper(self, value: float) -> None:
        self._range.upper = value
        self.notifyObservers()

    def setRangeAndLimits(self, interval: Interval[float]) -> None:
        self._range = interval.copy()
        self._limits = interval.copy()
        self.notifyObservers()
