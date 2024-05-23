from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import RealArrayType

from ..product import ProductRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExposureMap:
    pixel_width_m: float
    pixel_height_m: float
    center_x_m: float
    center_y_m: float
    counts: RealArrayType

    @property
    def pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(self.pixel_width_m, self.pixel_height_m)


class ExposureAnalyzer:

    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def analyze(self, itemIndex: int) -> ExposureMap:
        item = self._repository[itemIndex]
        objectItem = item.getObject()
        object_ = objectItem.getObject()

        counts = numpy.zeros_like(object_.array, dtype=float)  # FIXME

        return ExposureMap(
            pixel_width_m=object_.pixelWidthInMeters,
            pixel_height_m=object_.pixelHeightInMeters,
            center_x_m=object_.centerXInMeters,
            center_y_m=object_.centerYInMeters,
            counts=counts,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveResult(self, filePath: Path, result: ExposureMap) -> None:
        numpy.savez(
            filePath,
            'pixel_height_m',
            result.pixel_height_m,
            'pixel_width_m',
            result.pixel_width_m,
            'center_x_m',
            result.center_x_m,
            'center_y_m',
            result.center_y_m,
            'counts',
            result.counts,
        )
