import numpy

from ptychodus.api.scan import ScanBoundingBox, ScanPoint


class ScanBoundingBoxBuilder:
    def __init__(self) -> None:
        self._xmin_m = +numpy.inf
        self._xmax_m = -numpy.inf
        self._ymin_m = +numpy.inf
        self._ymax_m = -numpy.inf

    def hull(self, point: ScanPoint) -> None:
        if point.position_x_m < self._xmin_m:
            self._xmin_m = point.position_x_m

        if self._xmax_m < point.position_x_m:
            self._xmax_m = point.position_x_m

        if point.position_y_m < self._ymin_m:
            self._ymin_m = point.position_y_m

        if self._ymax_m < point.position_y_m:
            self._ymax_m = point.position_y_m

    def get_bounding_box(self) -> ScanBoundingBox | None:
        is_empty_x = self._xmax_m < self._xmin_m
        is_empty_y = self._ymax_m < self._ymin_m

        if is_empty_x or is_empty_y:
            return None

        return ScanBoundingBox(
            minimum_x_m=self._xmin_m,
            maximum_x_m=self._xmax_m,
            minimum_y_m=self._ymin_m,
            maximum_y_m=self._ymax_m,
        )
