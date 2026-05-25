from __future__ import annotations
from dataclasses import replace
import logging

from ptychodus.api.metrics import FourierRingCorrelation, compute_fourier_ring_correlation
from ptychodus.api.object import align_objects
from ptychodus.api.reconstructor import ReconstructionAmbiguities

from ..product import ProductRepository

__all__ = ['FourierRingCorrelation', 'FourierRingCorrelator']

logger = logging.getLogger(__name__)


class FourierRingCorrelator:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def correlate(self, product_index_1: int, product_index_2: int) -> FourierRingCorrelation:
        product1 = self._repository[product_index_1].get_product()
        product2 = self._repository[product_index_2].get_product()

        aligned_object2 = align_objects(product1.object_, product2.object_)
        aligned_product2 = replace(product2, object_=aligned_object2)

        ambiguities = ReconstructionAmbiguities.estimate(aligned_product2, reference=product1)
        standardized_product2 = ambiguities.standardize_product(aligned_product2)

        object1 = product1.object_
        object2 = standardized_product2.object_

        if object1.num_layers > 1 or object2.num_layers > 1:
            logger.warning('FRC flattens multi-layer objects; per-layer FRC is not implemented.')

        array1 = object1.get_layers_flattened()
        array2 = object2.get_layers_flattened()

        # TODO apply soft-edged mask
        pixel_geometry = object1.get_pixel_geometry()

        return compute_fourier_ring_correlation(
            array1,
            array2,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )
