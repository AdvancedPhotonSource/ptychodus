from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectCenter
from ptychodus.api.observer import Observable
from ptychodus.api.visualization import RealArrayType

from ..reconstructor import DiffractionPatternPositionMatcher
from .barycentric import BarycentricArrayStitcher

__all__ = [
    'STXMSimulator',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STXMData:
    intensity: RealArrayType
    pixel_geometry: PixelGeometry | None
    center: ObjectCenter | None


class STXMSimulator(Observable):
    def __init__(self, data_matcher: DiffractionPatternPositionMatcher) -> None:
        super().__init__()
        self._data_matcher = data_matcher

        self._product_index = -1
        self._product_data: STXMData | None = None

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._product_data = None
            self.notify_observers()

    def get_product_name(self) -> str:
        return self._data_matcher.get_product_name(self._product_index)

    def simulate(self) -> None:
        reconstruct_input = self._data_matcher.match_diffraction_patterns_with_positions(
            self._product_index
        )
        product = reconstruct_input.product
        object_geometry = product.object_.get_geometry()
        object_shape = object_geometry.height_px, object_geometry.width_px

        stitcher = BarycentricArrayStitcher[numpy.double](
            upper=numpy.zeros(object_shape),
            lower=numpy.zeros(object_shape),
        )

        for pattern, scan_point in zip(reconstruct_input.patterns, product.positions):
            pattern_counts = pattern.sum()
            probe_intensity = product.probe.get_intensity()  # FIXME OPR
            probe_profile = probe_intensity / numpy.sqrt(numpy.sum(numpy.abs(probe_intensity) ** 2))
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            stitcher.add_patch(
                object_point.position_x_px,
                object_point.position_y_px,
                pattern_counts * probe_profile,
                numpy.ones_like(probe_profile),
            )

        self._product_data = STXMData(
            intensity=stitcher.stitch(),  # FIXME
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
        )
        self.notify_observers()

    def get_data(self) -> STXMData:
        if self._product_data is None:
            raise ValueError('No simulated data!')

        return self._product_data

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_data(self, file_path: Path) -> None:
        if self._product_data is None:
            raise ValueError('No simulated data!')

        contents: dict[str, Any] = {
            'intensity': self._product_data.intensity,
        }

        pixel_geometry = self._product_data.pixel_geometry

        if pixel_geometry is not None:
            contents['pixel_height_m'] = pixel_geometry.height_m
            contents['pixel_width_m'] = pixel_geometry.width_m

        center = self._product_data.center

        if center is not None:
            contents['center_x_m'] = center.position_x_m
            contents['center_y_m'] = center.position_y_m

        numpy.savez(file_path, **contents)
