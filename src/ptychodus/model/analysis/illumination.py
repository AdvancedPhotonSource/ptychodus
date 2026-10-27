from __future__ import annotations
import logging
import time

from ptychodus.api.illumination import IlluminationMap, compute_illumination_map

from ..product import ProductRepository


__all__ = [
    'IlluminationMap',
    'IlluminationMapper',
]

logger = logging.getLogger(__name__)


class IlluminationMapper:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def map(self, product_index: int) -> IlluminationMap:
        item = self._repository[product_index]
        product = item.get_product()

        # Dose follows the incident beam, so positions are weighted by their I0 when it
        # was recorded and equally otherwise; the detector totals would fold the
        # sample's own absorption into the dose.
        logger.info('Computing illumination map...')
        tic = time.perf_counter()
        result = compute_illumination_map(
            product,
            probe_photon_counts_by_index=product.probe_positions.get_probe_photon_counts_by_index(),
        )
        toc = time.perf_counter()
        logger.info(f'Computed illumination map in {toc - tic:.4f} seconds.')

        return result
