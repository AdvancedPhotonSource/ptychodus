from __future__ import annotations
import logging
import time

from ptychodus.api.analyze import (
    ProbeComparisonReport,
    ProbeMetricsReport,
    compute_probe_comparison_report,
    compute_probe_metrics_report,
)

from ..product import ProductRepository

__all__ = [
    'ProbeAnalyzer',
    'ProbeComparisonReport',
    'ProbeMetricsReport',
]

logger = logging.getLogger(__name__)


class ProbeAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def compute_metrics(self, product_index: int) -> ProbeMetricsReport:
        product = self._repository[product_index].get_product()

        logger.info('Computing probe metrics...')
        tic = time.perf_counter()
        report = compute_probe_metrics_report(product)
        toc = time.perf_counter()
        logger.info(f'Computed probe metrics in {toc - tic:.4f} seconds.')

        return report

    def compare(self, reference_index: int, test_index: int) -> ProbeComparisonReport:
        reference = self._repository[reference_index].get_product()
        test = self._repository[test_index].get_product()

        logger.info('Comparing probes...')
        tic = time.perf_counter()
        report = compute_probe_comparison_report(reference, test)
        toc = time.perf_counter()
        logger.info(f'Compared probes in {toc - tic:.4f} seconds.')

        return report
