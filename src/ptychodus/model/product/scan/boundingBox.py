import numpy

from ptychodus.api.scan import ScanBoundingBox, ScanPoint


class ScanBoundingBoxBuilder:
    def __init__(self) -> None:
        self._minimumXInMeters = +numpy.inf
        self._maximumXInMeters = -numpy.inf
        self._minimumYInMeters = +numpy.inf
        self._maximumYInMeters = -numpy.inf

    def hull(self, point: ScanPoint) -> None:
        if point.position_x_m < self._minimumXInMeters:
            self._minimumXInMeters = point.position_x_m

        if self._maximumXInMeters < point.position_x_m:
            self._maximumXInMeters = point.position_x_m

        if point.position_y_m < self._minimumYInMeters:
            self._minimumYInMeters = point.position_y_m

        if self._maximumYInMeters < point.position_y_m:
            self._maximumYInMeters = point.position_y_m

    def getBoundingBox(self) -> ScanBoundingBox | None:
        isEmptyX = self._maximumXInMeters < self._minimumXInMeters
        isEmptyY = self._maximumYInMeters < self._minimumYInMeters

        if isEmptyX or isEmptyY:
            return None

        return ScanBoundingBox(
            minimum_x_m=self._minimumXInMeters,
            maximum_x_m=self._maximumXInMeters,
            minimum_y_m=self._minimumYInMeters,
            maximum_y_m=self._maximumYInMeters,
        )
