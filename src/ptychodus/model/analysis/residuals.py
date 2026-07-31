from __future__ import annotations

import logging
import time

from ptychodus.api.metrics import (
    ReconstructionResiduals,
    compute_reconstruction_residuals,
)

from ..product import ProductRepository

__all__ = ['ReconstructionResiduals', 'ResidualAnalyzer']

logger = logging.getLogger(__name__)


class ResidualAnalyzer:
    def __init__(
        self,
        repository: ProductRepository,
    ) -> None:
        self._repository = repository

    def analyze(self, product_index: int) -> ReconstructionResiduals:
        item = self._repository[product_index]
        dataset = item.get_dataset()

        if dataset is None:
            raise RuntimeError(
                f'Product "{item.get_name()}" has no associated diffraction dataset.'
            )

        product = item.get_product()
        recon_input = dataset.get_assembled_data().prepare_reconstruct_input(product)

        logger.info('Computing reconstruction residuals...')
        tic = time.perf_counter()
        result = compute_reconstruction_residuals(
            product=recon_input.product,
            measured_patterns=recon_input.diffraction_patterns,
            bad_pixels=recon_input.bad_pixels,
        )
        toc = time.perf_counter()
        logger.info(f'Computed reconstruction residuals in {toc - tic:.4f} seconds.')

        return result
