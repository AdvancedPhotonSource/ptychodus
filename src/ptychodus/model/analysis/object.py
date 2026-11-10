from __future__ import annotations
import logging
import time

from ptychodus.api.analyze import (
    ObjectComparisonReport,
    ObjectMetricsReport,
    compute_object_comparison_report,
    compute_object_metrics_report,
)
from ptychodus.api.metrics import SpectralSNRConvention

from ..product import ProductRepository

__all__ = [
    'ObjectAnalyzer',
    'ObjectComparisonReport',
    'ObjectMetricsReport',
    'SpectralSNRConvention',
]

logger = logging.getLogger(__name__)


class ObjectAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def compute_metrics(self, product_index: int) -> ObjectMetricsReport:
        product = self._repository[product_index].get_product()

        logger.info('Computing object metrics...')
        tic = time.perf_counter()
        report = compute_object_metrics_report(product)
        toc = time.perf_counter()
        logger.info(f'Computed object metrics in {toc - tic:.4f} seconds.')

        return report

    def compare(
        self,
        reference_index: int,
        test_index: int,
        *,
        convention: SpectralSNRConvention = SpectralSNRConvention.HALF_SPLIT,
    ) -> ObjectComparisonReport:
        reference = self._repository[reference_index].get_product()
        test = self._repository[test_index].get_product()

        logger.info('Comparing objects...')
        tic = time.perf_counter()
        report = compute_object_comparison_report(reference, test, convention=convention)
        toc = time.perf_counter()
        logger.info(f'Compared objects in {toc - tic:.4f} seconds.')

        return report
