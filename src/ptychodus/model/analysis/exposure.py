from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.visualization import RealArrayType

from ..product import ProductRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExposureMap:
    pixel_geometry: PixelGeometry | None
    center_x_m: float
    center_y_m: float
    counts: RealArrayType


class ExposureAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def analyze(self, itemIndex: int) -> ExposureMap:
        item = self._repository[itemIndex]
        objectItem = item.getObject()
        object_ = objectItem.getObject()

        counts = numpy.zeros_like(object_.getArray(), dtype=float)  # FIXME

        return ExposureMap(
            pixel_geometry=object_.getPixelGeometry(),
            center_x_m=object_.centerXInMeters,
            center_y_m=object_.centerYInMeters,
            counts=counts,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveResult(self, file_path: Path, result: ExposureMap) -> None:
        contents: dict[str, Any] = {
            'center_x_m': result.center_x_m,
            'center_y_m': result.center_y_m,
            'counts': result.counts,
        }

        pixel_geometry = result.pixel_geometry

        if pixel_geometry is not None:
            contents['pixel_height_m'] = pixel_geometry.heightInMeters
            contents['pixel_width_m'] = pixel_geometry.widthInMeters

        numpy.savez(file_path, **contents)
