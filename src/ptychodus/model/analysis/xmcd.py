from __future__ import annotations
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.object import align_objects
from ptychodus.api.observer import Observable
from ptychodus.api.reconstructor import ReconstructionAmbiguities
from ptychodus.api.xmcd import XMCDResult, estimate_xmcd

from ..product import ProductRepository

logger = logging.getLogger(__name__)


class XMCDAnalyzer(Observable):
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository

        self._lcp_product_index = -1
        self._rcp_product_index = -1
        self._result: XMCDResult | None = None

    def set_lcp_product(self, lcirc_product_index: int) -> None:
        if self._lcp_product_index != lcirc_product_index:
            self._lcp_product_index = lcirc_product_index
            self.notify_observers()

    def get_lcp_product(self) -> int:
        return self._lcp_product_index

    def set_rcp_product(self, rcirc_product_index: int) -> None:
        if self._rcp_product_index != rcirc_product_index:
            self._rcp_product_index = rcirc_product_index
            self.notify_observers()

    def get_rcp_product(self) -> int:
        return self._rcp_product_index

    def analyze(self) -> None:
        lcp_product = self._repository[self._lcp_product_index].get_product()
        rcp_product = self._repository[self._rcp_product_index].get_product()

        if lcp_product.object_.num_layers > 1 or rcp_product.object_.num_layers > 1:
            logger.warning('XMCD flattens multi-layer objects; per-layer XMCD is not implemented.')

        aligned_lcp_object = align_objects(rcp_product.object_, lcp_product.object_)
        aligned_lcp_product = replace(lcp_product, object_=aligned_lcp_object)

        ambiguities = ReconstructionAmbiguities.estimate(aligned_lcp_product, reference=rcp_product)
        standardized_lcp_product = ambiguities.standardize_product(aligned_lcp_product)

        self._result = estimate_xmcd(
            rcp_object=rcp_product.object_,
            lcp_object_aligned=standardized_lcp_product.object_,
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

    def save_data(self, file_path: Path) -> None:  # FIXME rethink
        if self._result is None:
            raise ValueError('No analyzed data!')

        structural_object = self._result.structural_object
        magnetic_object = self._result.magnetic_object
        pixel_geometry = structural_object.get_pixel_geometry()
        center = structural_object.get_center()

        contents: dict[str, Any] = {
            'structural_object': structural_object.get_array(),
            'magnetic_object': magnetic_object.get_array(),
            'pixel_height_m': pixel_geometry.height_m,
            'pixel_width_m': pixel_geometry.width_m,
            'center_x_m': center.coordinate_x_m,
            'center_y_m': center.coordinate_y_m,
        }

        numpy.savez_compressed(file_path, allow_pickle=False, **contents)
