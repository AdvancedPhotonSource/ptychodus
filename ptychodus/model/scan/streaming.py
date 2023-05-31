from bisect import bisect
from collections import defaultdict
from collections.abc import Sequence
from decimal import Decimal
from statistics import median

from ...api.scan import ScanPoint, TabularScan


class PositionStream:

    def __init__(self) -> None:
        self.valuesInMeters: list[float] = list()
        self.timeStamps: list[float] = list()

    def clear(self) -> None:
        self.valuesInMeters.clear()
        self.timeStamps.clear()

    def assemble(self, valuesInMeters: Sequence[float], timeStamps: Sequence[float]) -> None:
        self.valuesInMeters.extend(valuesInMeters)
        self.timeStamps.extend(timeStamps)

    def getMedianPositions(self, arrayTimeStampDict: dict[int, float]) -> dict[int, float]:
        valuesSeqMap: dict[int, list[float]] = defaultdict(list[float])
        arrayIndexList: list[int] = list()
        arrayTimeStampList: list[float] = list()

        for index, timeStamp in sorted(arrayTimeStampDict.items()):
            arrayIndexList.append(index)
            arrayTimeStampList.append(timeStamp)

        for valueInMeters, timeStamp in zip(self.valuesInMeters, self.timeStamps):
            index = bisect(arrayTimeStampList, timeStamp)

            try:
                arrayIndex = arrayIndexList[index]
            except IndexError:
                break
            else:
                valuesSeqMap[arrayIndex].append(valueInMeters)

        return {index: median(values) for index, values in valuesSeqMap.items()}


class StreamingScanBuilder:

    def __init__(self) -> None:
        self._streamX = PositionStream()
        self._streamY = PositionStream()
        self._arrayTimeStamps: dict[int, float] = dict()

    def reset(self) -> None:
        self._streamX.clear()
        self._streamY.clear()
        self._arrayTimeStamps.clear()

    def insertArrayTimeStamp(self, arrayIndex: int, timeStamp: float) -> None:
        self._arrayTimeStamps[arrayIndex] = timeStamp

    def assembleScanPositionsX(self, valuesInMeters: Sequence[float],
                               timeStamps: Sequence[float]) -> None:
        self._streamX.assemble(valuesInMeters, timeStamps)

    def assembleScanPositionsY(self, valuesInMeters: Sequence[float],
                               timeStamps: Sequence[float]) -> None:
        self._streamY.assemble(valuesInMeters, timeStamps)

    def build(self) -> TabularScan:
        posX = self._streamX.getMedianPositions(self._arrayTimeStamps)
        posY = self._streamY.getMedianPositions(self._arrayTimeStamps)

        arrayIndexSet = set(self._arrayTimeStamps) & set(posX) & set(posY)
        pointMap: dict[int, ScanPoint] = dict()

        for index in arrayIndexSet:
            pointMap[index] = ScanPoint(
                x=Decimal.from_float(posX[index]),
                y=Decimal.from_float(posY[index]),
            )

        return TabularScan(pointMap)
