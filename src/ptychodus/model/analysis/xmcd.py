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
from ptychodus.api.typing import ComplexArrayType

from ..product import ProductRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XMCDResult:
    polar_difference: ComplexArrayType
    polar_sum: ComplexArrayType
    polar_ratio: ComplexArrayType
    pixel_geometry: PixelGeometry
    center: ObjectCenter


class XMCDAnalyzer(Observable):
    # TODO feature request: want ability to align/add reconstructed slices
    #      of repeat scans for each polarization separately to improve statistics

    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository

        self._lcirc_product_index = -1
        self._rcirc_product_index = -1
        self._result: XMCDResult | None = None

    def set_lcirc_product(self, lcirc_product_index: int) -> None:
        if self._lcirc_product_index != lcirc_product_index:
            self._lcirc_product_index = lcirc_product_index
            self._lcirc_product_data = None
            self.notify_observers()

    def get_lcirc_product(self) -> int:
        return self._lcirc_product_index

    def get_lcirc_product_name(self) -> str:
        lcirc_product = self._repository[self._lcirc_product_index]
        return lcirc_product.get_name()

    def set_rcirc_product(self, rcirc_product_index: int) -> None:
        if self._rcirc_product_index != rcirc_product_index:
            self._rcirc_product_index = rcirc_product_index
            self._rcirc_product_data = None
            self.notify_observers()

    def get_rcirc_product(self) -> int:
        return self._rcirc_product_index

    def get_rcirc_product_name(self) -> str:
        rcirc_product = self._repository[self._rcirc_product_index]
        return rcirc_product.get_name()

    def analyze(self) -> None:
        lcirc_object = self._repository[self._lcirc_product_index].get_object_item().get_object()
        rcirc_object = self._repository[self._rcirc_product_index].get_object_item().get_object()

        lcirc_object_geometry = lcirc_object.get_geometry()
        rcirc_object_geometry = rcirc_object.get_geometry()

        if lcirc_object_geometry.width_px != rcirc_object_geometry.width_px:
            raise ValueError('Object width mismatch!')

        if lcirc_object_geometry.height_px != rcirc_object_geometry.height_px:
            raise ValueError('Object height mismatch!')

        if lcirc_object_geometry.pixel_width_m != rcirc_object_geometry.pixel_width_m:
            raise ValueError('Object pixel width mismatch!')

        if lcirc_object_geometry.pixel_height_m != rcirc_object_geometry.pixel_height_m:
            raise ValueError('Object pixel height mismatch!')

        # TODO align lcirc_array/rcirc_array
        lcirc_amp = numpy.absolute(lcirc_object.get_layers_flattened())
        rcirc_amp = numpy.absolute(rcirc_object.get_layers_flattened())

        ratio = numpy.divide(lcirc_amp, rcirc_amp)
        product = numpy.multiply(lcirc_amp, rcirc_amp)

        polar_difference = numpy.log(ratio, out=numpy.zeros_like(ratio), where=(ratio > 0.0))
        polar_sum = numpy.log(product, out=numpy.zeros_like(product), where=(product > 0.0))
        polar_ratio = numpy.divide(
            polar_difference,
            polar_sum,
            out=numpy.zeros_like(polar_sum),
            where=(polar_sum > 0.0),
        )

        self._result = XMCDResult(
            polar_difference=polar_difference,
            polar_sum=polar_sum,
            polar_ratio=polar_ratio,
            pixel_geometry=rcirc_object.get_pixel_geometry(),
            center=rcirc_object.get_center(),
        )
        self.notify_observers()

    def get_result(self) -> XMCDResult:
        if self._result is None:
            raise ValueError('No analyzed data!')

        return self._result

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_data(self, file_path: Path) -> None:
        if self._result is None:
            raise ValueError('No analyzed data!')

        contents: dict[str, Any] = {
            'polar_difference': self._result.polar_difference,
            'polar_sum': self._result.polar_sum,
            'polar_ratio': self._result.polar_ratio,
            'pixel_height_m': self._result.pixel_geometry.height_m,
            'pixel_width_m': self._result.pixel_geometry.width_m,
            'center_x_m': self._result.center.position_x_m,
            'center_y_m': self._result.center.position_y_m,
        }

        numpy.savez_compressed(file_path, allow_pickle=False, **contents)
