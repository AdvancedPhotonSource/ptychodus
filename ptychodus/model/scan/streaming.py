from decimal import Decimal
from statistics import median

from ...api.scan import ScanPoint, TabularScan
from .factory import ScanRepositoryItemFactory
from .repositoryItem import ScanRepositoryItem


class PositionStream:

    def __init__(self) -> None:
        self.valuesInMeters: list[float] = list()
        self.arrayIndexes: list[int] = list()

    def clear(self) -> None:
        self.valuesInMeters.clear()
        self.arrayIndexes.clear()

    def assemble(self, arrayIndexes: list[int], valuesInMeters: list[float]) -> None:
        self.arrayIndexes.extend(arrayIndexes)
        self.valuesInMeters.extend(valuesInMeters)

    def __getitem__(self, index: int) -> float:
        return median(self.valuesInMeters[self.arrayIndexes == index])


class StreamingScanBuilder:

    def __init__(self, factory: ScanRepositoryItemFactory) -> None:
        self._factory = factory
        self._streamX = PositionStream()
        self._streamY = PositionStream()

    def reset(self) -> None:
        self._streamX.clear()
        self._streamY.clear()

    def assembleScanPositionsX(self, arrayIndexes: list[int], valuesInMeters: list[float]) -> None:
        self._streamX.assemble(arrayIndexes, valuesInMeters)

    def assembleScanPositionsY(self, arrayIndexes: list[int], valuesInMeters: list[float]) -> None:
        self._streamY.assemble(arrayIndexes, valuesInMeters)

    def build(self) -> ScanRepositoryItem:
        arrayIndexes = list(set(self._streamX.arrayIndexes) & set(self._streamY.arrayIndexes))
        pointMap: dict[int, ScanPoint] = dict()

        for index in sorted(arrayIndexes):
            pointMap[index] = ScanPoint(
                x=Decimal(repr(self._streamX[index])),
                y=Decimal(repr(self._streamY[index])),
            )

        scan = TabularScan('Stream', pointMap)
        return self._factory.createTabularItem(scan, None)
