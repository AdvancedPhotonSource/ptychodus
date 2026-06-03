from __future__ import annotations

from ptychodus.api.metrics import (
    FourierRingCorrelation,
    ObjectComparison,
    compute_fourier_ring_correlation,
)

from ..product import ProductRepository

__all__ = ['FourierRingCorrelation', 'FourierRingCorrelator']


class FourierRingCorrelator:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def correlate(self, product_index_1: int, product_index_2: int) -> FourierRingCorrelation:
        product1 = self._repository[product_index_1].get_product()
        product2 = self._repository[product_index_2].get_product()

        comparison = ObjectComparison.from_products(reference=product1, test=product2)

        # TODO apply soft-edged mask
        return compute_fourier_ring_correlation(
            comparison.reference_complex,
            comparison.test_complex,
            pixel_width_m=comparison.pixel_geometry.width_m,
            pixel_height_m=comparison.pixel_geometry.height_m,
        )
