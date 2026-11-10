from __future__ import annotations
import logging
import time

from ptychodus.api.analyze import (
    ProbePositionsComparisonReport,
    ProbePositionsMetricsReport,
    compute_probe_positions_comparison_report,
    compute_probe_positions_metrics_report,
)

from ..product import ProductRepository

__all__ = [
    'ProbePositionsAnalyzer',
    'ProbePositionsComparisonReport',
    'ProbePositionsMetricsReport',
]

logger = logging.getLogger(__name__)


class ProbePositionsAnalyzer:
    """Reports on a product's probe positions, addressed by product index.

    The probe-positions repository presents the product repository's items in the same
    order, so one index names a product in either.
    """

    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def compute_metrics(self, product_index: int) -> ProbePositionsMetricsReport:
        product = self._repository[product_index].get_product()

        logger.info('Computing probe position metrics...')
        tic = time.perf_counter()
        report = compute_probe_positions_metrics_report(product)
        toc = time.perf_counter()
        logger.info(f'Computed probe position metrics in {toc - tic:.4f} seconds.')

        return report

    def compare(self, reference_index: int, test_index: int) -> ProbePositionsComparisonReport:
        reference = self._repository[reference_index].get_product()
        test = self._repository[test_index].get_product()

        logger.info('Comparing probe positions...')
        tic = time.perf_counter()
        report = compute_probe_positions_comparison_report(reference, test)
        toc = time.perf_counter()
        logger.info(f'Compared probe positions in {toc - tic:.4f} seconds.')

        return report
