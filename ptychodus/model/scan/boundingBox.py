import numpy

from ...api.geometry import Point2D
from ...api.scan import ScanBoundingBox, ScanPoint


class ScanBoundingBoxBuilder:

    def __init__(self) -> None:
        self._minimumXInMeters = +numpy.inf
        self._maximumXInMeters = -numpy.inf
        self._minimumYInMeters = +numpy.inf
        self._maximumYInMeters = -numpy.inf

    def hull(self, point: ScanPoint) -> None:
        if point.positionXInMeters < self._minimumXInMeters:
            self._minimumXInMeters = point.positionXInMeters

        if self._maximumXInMeters < point.positionXInMeters:
            self._maximumXInMeters = point.positionXInMeters

        if point.positionYInMeters < self._minimumYInMeters:
            self._minimumYInMeters = point.positionYInMeters

        if self._maximumYInMeters < point.positionYInMeters:
            self._maximumYInMeters = point.positionYInMeters

    def getMidpointInMeters(self) -> Point2D:
        return Point2D(
            x=0.5 * (self._minimumXInMeters + self._maximumXInMeters),
            y=0.5 * (self._minimumYInMeters + self._maximumYInMeters),
        )

    def getBoundingBox(self) -> ScanBoundingBox | None:
        isEmptyX = (self._maximumXInMeters < self._minimumXInMeters)
        isEmptyY = (self._maximumYInMeters < self._minimumYInMeters)

        if isEmptyX or isEmptyY:
            return None

        return ScanBoundingBox(
            minimumXInMeters=self._minimumXInMeters,
            maximumXInMeters=self._maximumXInMeters,
            minimumYInMeters=self._minimumYInMeters,
            maximumYInMeters=self._maximumYInMeters,
        )
