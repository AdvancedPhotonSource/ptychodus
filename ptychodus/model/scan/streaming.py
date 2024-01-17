from __future__ import annotations
from bisect import bisect
from collections import defaultdict
from collections.abc import Sequence
from statistics import median
import logging

from ...api.scan import Scan, ScanPoint

logger = logging.getLogger(__name__)

# TODO class PositionStream:
# TODO
# TODO     def __init__(self) -> None:
# TODO         self._indexes: list[int] = list()
# TODO         self._positions: list[float] = list()
# TODO         self._minimum = +numpy.inf
# TODO         self._maximum = -numpy.inf
# TODO         self._uniqueness = 0
# TODO
# TODO     @property
# TODO     def minimum(self) -> float:
# TODO         return self._minimum
# TODO
# TODO     @property
# TODO     def maximum(self) -> float:
# TODO         return self._maximum
# TODO
# TODO     def append(self, index: int, position: float) -> None:
# TODO         lastIndex = self._indexes[-1]
# TODO
# TODO         if index > lastIndex:
# TODO             self._uniqueness = 0
# TODO             self._indexes.append(index)
# TODO             self._positions.append(position)
# TODO         elif index == lastIndex:  # online mean
# TODO             self._uniqueness += 1
# TODO             self._positions[-1] += (position - self._positions[-1]) / self._uniqueness
# TODO         else:  # index < lastIndex
# TODO             logger.warning(f'Discarding non-monotonic {index=}!')
# TODO
# TODO         if position < self._minimum:
# TODO             self._minimum = position
# TODO
# TODO         if position > self._maximum:
# TODO             self._maximum = position


class PositionStream:  # FIXME clean up

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

    def build(self) -> Scan:
        posX = self._streamX.getMedianPositions(self._arrayTimeStamps)
        posY = self._streamY.getMedianPositions(self._arrayTimeStamps)

        arrayIndexSet = set(self._arrayTimeStamps) & set(posX) & set(posY)
        pointList: list[ScanPoint] = list()

        for index in arrayIndexSet:
            point = ScanPoint(index, posX[index], posY[index])
            pointList.append(point)

        return Scan(pointList)
