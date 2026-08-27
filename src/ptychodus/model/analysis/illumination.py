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

        probe_photon_counts_by_index: dict[int, float] | None = None
        dataset = item.get_dataset()
        if dataset is not None:
            assembled = dataset.get_assembled_data()
            probe_photon_counts_by_index = {
                int(idx): float(counts)
                for idx, counts in zip(assembled.get_indexes(), assembled.get_probe_photon_counts())
            }

        logger.info('Computing illumination map...')
        tic = time.perf_counter()
        result = compute_illumination_map(
            product, probe_photon_counts_by_index=probe_photon_counts_by_index
        )
        toc = time.perf_counter()
        logger.info(f'Computed illumination map in {toc - tic:.4f} seconds.')

        return result
