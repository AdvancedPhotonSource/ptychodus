from __future__ import annotations

import logging
import time

from ptychodus.api.metrics import (
    FourierRingCorrelation,
    ObjectComparison,
    compute_fourier_ring_correlation,
)

from ..product import ProductRepository

__all__ = ['FourierRingCorrelation', 'FourierRingCorrelator']

logger = logging.getLogger(__name__)


class FourierRingCorrelator:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def correlate(self, product_index_1: int, product_index_2: int) -> FourierRingCorrelation:
        product1 = self._repository[product_index_1].get_product()
        product2 = self._repository[product_index_2].get_product()

        comparison = ObjectComparison.from_products(reference=product1, test=product2)

        # TODO apply soft-edged mask
        logger.info('Computing Fourier ring correlation...')
        tic = time.perf_counter()
        result = compute_fourier_ring_correlation(
            comparison.reference_complex,
            comparison.test_complex,
            pixel_width_m=comparison.pixel_geometry.width_m,
            pixel_height_m=comparison.pixel_geometry.height_m,
        )
        toc = time.perf_counter()
        logger.info(f'Computed Fourier ring correlation in {toc - tic:.4f} seconds.')

        return result
