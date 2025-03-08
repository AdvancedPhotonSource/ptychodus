from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectCenter
from ptychodus.api.visualization import RealArrayType

from ..product import ProductRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExposureMap:
    pixel_geometry: PixelGeometry | None
    center: ObjectCenter | None
    counts: RealArrayType


class ExposureAnalyzer:  # XXX
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def analyze(self, itemIndex: int) -> ExposureMap:
        item = self._repository[itemIndex]
        objectItem = item.getObject()
        object_ = objectItem.getObject()

        counts = numpy.zeros_like(object_.getArray(), dtype=float)  # FIXME

        return ExposureMap(
            pixel_geometry=object_.getPixelGeometry(),
            center=object_.getCenter(),
            counts=counts,
        )

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def saveResult(self, file_path: Path, result: ExposureMap) -> None:
        contents: dict[str, Any] = {
            'counts': result.counts,
        }

        pixel_geometry = result.pixel_geometry

        if pixel_geometry is not None:
            contents['pixel_height_m'] = pixel_geometry.heightInMeters
            contents['pixel_width_m'] = pixel_geometry.widthInMeters

        center = result.center

        if center is not None:
            contents['center_x_m'] = center.positionXInMeters
            contents['center_y_m'] = center.positionYInMeters

        numpy.savez(file_path, **contents)
