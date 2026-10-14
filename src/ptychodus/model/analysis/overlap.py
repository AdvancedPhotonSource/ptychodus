from __future__ import annotations
import logging
import time

from ptychodus.api.illumination import ProbeOverlapMetrics, compute_probe_overlap

from ..product import ProductRepository


__all__ = [
    'ProbeOverlapAnalyzer',
    'ProbeOverlapMetrics',
]

logger = logging.getLogger(__name__)


class ProbeOverlapAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def analyze(self, product_index: int) -> ProbeOverlapMetrics:
        # Unlike the illumination map, overlap is geometry rather than photometry, so the
        # per-scan-index photon counts from the diffraction dataset are not needed here.
        product = self._repository[product_index].get_product()

        logger.info('Computing probe overlap...')
        tic = time.perf_counter()
        result = compute_probe_overlap(product)
        toc = time.perf_counter()
        logger.info(f'Computed probe overlap in {toc - tic:.4f} seconds.')

        return result
