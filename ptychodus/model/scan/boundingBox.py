import numpy

from ...api.geometry import Box2D, Interval, Point2D
from ...api.scan import ScanPoint


class ScanBoundingBox:

    def __init__(self) -> None:
        self._xmin = +numpy.inf
        self._xmax = -numpy.inf
        self._ymin = +numpy.inf
        self._ymax = -numpy.inf

    def hull(self, point: ScanPoint) -> None:
        if point.positionXInMeters < self._xmin:
            self._xmin = point.positionXInMeters

        if self._xmax < point.positionXInMeters:
            self._xmax = point.positionXInMeters

        if point.positionYInMeters < self._ymin:
            self._ymin = point.positionYInMeters

        if self._ymax < point.positionYInMeters:
            self._ymax = point.positionYInMeters

    def getMidpointInMeters(self) -> Point2D:
        return Point2D(
            x=0.5 * (self._xmin + self._xmax),
            y=0.5 * (self._ymin + self._ymax),
        )

    def getBoundingBoxInMeters(self) -> Box2D | None:
        return Box2D(
            rangeX=Interval[float](self._xmin, self._xmax),
            rangeY=Interval[float](self._ymin, self._ymax),
        )
