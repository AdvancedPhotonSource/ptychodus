from __future__ import annotations
from collections.abc import Iterable
import sys

import numpy

from ...api.geometry import Box2D, Interval, Point2D
from ...api.scan import ScanPoint


class ScanMetrics:

    def __init__(self) -> None:
        self._xmin = +numpy.inf
        self._xmax = -numpy.inf
        self._ymin = +numpy.inf
        self._ymax = -numpy.inf
        self._lengthInMeters = 0.
        self._sizeInBytes = 0
        self._previousPoint: ScanPoint | None = None

    @classmethod
    def createFromPoints(cls, points: Iterable[ScanPoint]) -> ScanMetrics:
        metrics = cls()

        for point in points:
            metrics.process(point)

        return metrics

    def process(self, point: ScanPoint, *, notify=True) -> None:
        if point.positionXInMeters < self._xmin:
            self._xmin = point.positionXInMeters

        if self._xmax < point.positionXInMeters:
            self._xmax = point.positionXInMeters

        if point.positionYInMeters < self._ymin:
            self._ymin = point.positionYInMeters

        if self._ymax < point.positionYInMeters:
            self._ymax = point.positionYInMeters

        if self._previousPoint is None:
            self._previousPoint = point
        else:
            dx = point.positionXInMeters - self._previousPoint.positionXInMeters
            dy = point.positionYInMeters - self._previousPoint.positionYInMeters
            self._lengthInMeters += numpy.hypot(dx, dy)

        self._sizeInBytes += sys.getsizeof(point)

    def getMidpointInMeters(self) -> Point2D:
        return Point2D(
            x=0.5 * (self._xmin + self._xmax),
            y=0.5 * (self._ymin + self._ymax),
        )

    def getLengthInMeters(self) -> float:
        return self._lengthInMeters

    def getSizeInBytes(self) -> int:
        return self._sizeInBytes

    def getBoundingBoxInMeters(self) -> Box2D | None:
        return Box2D(
            rangeX=Interval[float](self._xmin, self._xmax),
            rangeY=Interval[float](self._ymin, self._ymax),
        )
